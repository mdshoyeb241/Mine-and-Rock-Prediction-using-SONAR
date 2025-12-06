import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve)

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import logging
from datetime import datetime

# Configure logging to track our model's training process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training_output.log')
logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')

class SonarMineDetector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        self.training_history = {}
        
        # Set up paths for model persistence
        self.model_dir = Path("trained_models")
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info("SonarMineDetector initialized with random_state=%d", random_state)

    def load_and_preprocess_data(self, csv_path):
        logger.info("Loading sonar data from %s", csv_path)
        
        try:
            # Load the CSV file without headers since sonar data typically doesn't include them
            data = pd.read_csv(csv_path, header=None)
            logger.info("Successfully loaded data with shape: %s", data.shape)
            
            # Validate data dimensions based on your specifications
            if data.shape != (208, 61):
                logger.warning("Expected shape (208, 61), got %s. Proceeding with loaded data.", data.shape)
            
            # Separate features (first 60 columns) from labels (last column)
            X = data.iloc[:, :-1].values  # All columns except the last one
            y = data.iloc[:, -1].values   # Last column contains the labels
            
            # Create meaningful feature names for interpretability
            self.feature_names = [f'Frequency_Band_{i+1}' for i in range(X.shape[1])]
            
            logger.info("Features extracted: %d features from %d samples", X.shape[1], X.shape[0])
            logger.info("Label distribution before encoding:")
            unique_labels, counts = np.unique(y, return_counts=True)
            for label, count in zip(unique_labels, counts):
                logger.info("  %s: %d samples (%.1f%%)", label, count, 100*count/len(y))
            
            # Convert string labels to numerical values
            # This is essential because scikit-learn requires numerical targets
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Log the encoding mapping for transparency
            logger.info("Label encoding mapping:")
            for original, encoded in zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)):
                logger.info("  '%s' -> %d", original, encoded)
            
            # Validate that we have both classes represented
            if len(np.unique(y_encoded)) < 2:
                raise ValueError("Dataset must contain both mine and rock samples")
            
            # Check for any missing or invalid values
            if np.any(np.isnan(X)):
                logger.warning("Found NaN values in features. Consider data cleaning.")
            
            # Log basic statistics about the features
            logger.info("Feature statistics:")
            logger.info("  Min value: %.4f", np.min(X))
            logger.info("  Max value: %.4f", np.max(X))
            logger.info("  Mean value: %.4f", np.mean(X))
            logger.info("  Std deviation: %.4f", np.std(X))
            
            return X, y_encoded
            
        except FileNotFoundError:
            logger.error("Could not find the CSV file at path: %s", csv_path)
            raise
        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            raise


    def prepare_features(self, X_train, X_test=None):
        logger.info("Standardizing features...")
        
        # Fit the scaler on training data only to prevent data leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        logger.info("Feature standardization completed:")
        logger.info("  Training set - Mean: %.4f, Std: %.4f", np.mean(X_train_scaled), np.std(X_train_scaled))
        
        if X_test is not None:
            # Apply the same transformation to test data
            X_test_scaled = self.scaler.transform(X_test)
            logger.info("  Test set - Mean: %.4f, Std: %.4f", np.mean(X_test_scaled), np.std(X_test_scaled))
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled

    def optimize_hyperparameters(self, X_train, y_train):
        logger.info("Starting hyperparameter optimization...")
        
        # Define parameter grid for grid search
        # We use a focused grid to balance performance with computational time
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create base Random Forest classifier
        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        # Perform grid search with cross-validation
        # We use 5-fold CV for robust evaluation given our limited dataset size
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='f1',  # F1 score balances precision and recall
            n_jobs=-1,  # Use all available CPU cores
            verbose=1
        )
        
        logger.info("Fitting grid search (this may take several minutes)...")
        grid_search.fit(X_train, y_train)
        
        logger.info("Hyperparameter optimization completed!")
        logger.info("Best parameters: %s", grid_search.best_params_)
        logger.info("Best cross-validation F1 score: %.4f", grid_search.best_score_)
        
        return grid_search.best_params_
    
    def train_model(self, X_train, y_train, optimize_params=True):
        logger.info("Training Random Forest classifier...")
        
        if optimize_params:
            best_params = self.optimize_hyperparameters(X_train, y_train)
        else:
            # Use reasonable default parameters
            best_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 5,
                'max_features': 'sqrt',
                'min_samples_leaf': 2
            }
            logger.info("Using default parameters: %s", best_params)
        
        # Create and train the final model
        self.model = RandomForestClassifier(
            **best_params,
            random_state=self.random_state,
            n_jobs=-1,
            oob_score=True  # Calculate out-of-bag score for additional validation
        )
        
        # Train the model
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = datetime.now() - start_time
        
        self.is_trained = True
        
        # Store training information
        self.training_history = {
            'parameters': best_params,
            'training_time': training_time.total_seconds(),
            'oob_score': self.model.oob_score_,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        logger.info("Model training completed in %.2f seconds", training_time.total_seconds())
        logger.info("Out-of-bag score: %.4f", self.model.oob_score_)
        logger.info("Model uses %d trees with %d features per sample", self.model.n_estimators, X_train.shape[1])

    def evaluate_model(self, X_test, y_test, detailed=True):
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate all important metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Confusion matrix provides detailed error analysis
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate additional meaningful metrics for military context
        false_positive_rate = fp / (fp + tn)  # Mistaking rocks for mines
        false_negative_rate = fn / (fn + tp)  # Missing actual mines
        specificity = tn / (tn + fp)  # Correctly identifying rocks
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'specificity': specificity,
            'confusion_matrix': cm,
            'oob_score': self.training_history.get('oob_score', 'N/A')
        }
        
        # Print detailed results
        print("\n" + "="*70)
        print("SONAR MINE DETECTION - MODEL EVALUATION RESULTS")
        print("="*70)
        print(f"Overall Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision (Mine):      {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall (Mine):         {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:              {f1:.4f} ({f1*100:.2f}%)")
        print(f"AUC-ROC Score:         {auc_score:.4f}")
        print(f"Out-of-Bag Score:      {results['oob_score']}")
        
        print(f"\nOperational Metrics:")
        print(f"False Positive Rate:   {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
        print(f"False Negative Rate:   {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
        print(f"Specificity (Rock):    {specificity:.4f} ({specificity*100:.2f}%)")
        
        print(f"\nConfusion Matrix:")
        print("                    Predicted")
        print("                 Rock    Mine")
        print(f"Actual Rock   [{tn:4d}]  [{fp:4d}]")
        print(f"       Mine   [{fn:4d}]  [{tp:4d}]")
        
        if detailed:
            print(f"\nDetailed Classification Report:")
            target_names = ['Rock', 'Mine']
            print(classification_report(y_test, y_pred, target_names=target_names))
            
            # Operational interpretation
            print(f"\n" + "="*70)
            print("OPERATIONAL INTERPRETATION")
            print("="*70)
            if false_negative_rate < 0.05:
                print("✅ EXCELLENT: Very low mine miss rate - high operational safety")
            elif false_negative_rate < 0.10:
                print("✅ GOOD: Low mine miss rate - acceptable operational risk")
            else:
                print("⚠️ CAUTION: Higher mine miss rate - consider adjusting threshold")
            
            if false_positive_rate < 0.10:
                print("✅ EXCELLENT: Low false alarm rate - efficient operations")
            elif false_positive_rate < 0.20:
                print("✅ GOOD: Moderate false alarm rate - manageable resource impact")
            else:
                print("⚠️ CAUTION: High false alarm rate - may waste operational resources")
        
        logger.info("Model evaluation completed")
        return results

    def analyze_feature_importance(self, top_n=15):
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing feature importance")
        
        logger.info("Analyzing feature importance...")
        
        # Get feature importances from the Random Forest
        importances = self.model.feature_importances_
        
        # Create DataFrame for easy analysis
        feature_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances,
            'Frequency_Band': range(1, len(self.feature_names) + 1)
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Sonar Frequency Bands:")
        print("-" * 60)
        for idx, (_, row) in enumerate(feature_df.head(top_n).iterrows(), 1):
            print(f"{idx:2d}. {row['Feature']}: {row['Importance']:.4f}")
        
        # Calculate cumulative importance
        cumulative_importance = np.cumsum(feature_df['Importance'].values)
        features_for_90_percent = np.argmax(cumulative_importance >= 0.9) + 1
        
        print(f"\nFeature Analysis Summary:")
        print(f"  • Top {features_for_90_percent} features explain 90% of importance")
        print(f"  • Most important band: {feature_df.iloc[0]['Feature']}")
        print(f"  • Least important band: {feature_df.iloc[-1]['Feature']}")
        
        return feature_df
    
    def predict_single_sample(self, features, return_confidence=True):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert input to numpy array and validate
        features = np.array(features).reshape(1, -1)
        
        if features.shape[1] != 60:
            raise ValueError(f"Expected 60 features, got {features.shape[1]}")
        
        # Apply the same standardization used during training
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction_encoded = self.model.predict(features_scaled)[0]
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        
        # Convert back to original label
        prediction_label = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        if return_confidence:
            confidence = np.max(prediction_proba)
            return prediction_label, confidence
        
        return prediction_label
    
    def save_model(self, model_name="sonar_mine_detector"):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        logger.info("Saving trained model system...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.model_dir / f"{model_name}_{timestamp}"
        
        # Save model components
        joblib.dump(self.model, f"{base_path}_model.pkl")
        joblib.dump(self.scaler, f"{base_path}_scaler.pkl")
        joblib.dump(self.label_encoder, f"{base_path}_encoder.pkl")

        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_type': 'RandomForestClassifier',
            'n_features': len(self.feature_names),
            'label_classes': list(self.label_encoder.classes_)
        }
        joblib.dump(metadata, f"{base_path}_metadata.pkl")
        
        logger.info("Model saved successfully:")
        logger.info("  Model: %s_model.pkl", base_path)
        logger.info("  Scaler: %s_scaler.pkl", base_path)
        logger.info("  Encoder: %s_encoder.pkl", base_path)
        logger.info("  Metadata: %s_metadata.pkl", base_path)
        
        return str(base_path)
    
    def load_model(self, model_path):
        logger.info("Loading trained model system from %s...", model_path)
        
        try:
            self.model = joblib.load(f"{model_path}_model.pkl")
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            self.label_encoder = joblib.load(f"{model_path}_encoder.pkl")
            
            # Load metadata if available
            try:
                metadata = joblib.load(f"{model_path}_metadata.pkl")
                self.feature_names = metadata['feature_names']
                self.training_history = metadata['training_history']
                logger.info("Model metadata loaded successfully")
            except FileNotFoundError:
                logger.warning("Model metadata not found, using defaults")
                self.feature_names = [f'Frequency_Band_{i+1}' for i in range(60)]
            
            self.is_trained = True
            logger.info("Model system loaded successfully")
            return self.model, self.scaler, self.label_encoder
            
        except FileNotFoundError as e:
            logger.error("Could not find model files: %s", str(e))
            raise

        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise

def model_training_implementation(csv_path="sonarall_data.csv"):
    print("Initializing submarine mine detection system...")
    print("This system will classify underwater objects as mines or rocks")
    print("based on their sonar signature patterns.\n")
    detector = SonarMineDetector(random_state=42)

    try:
        print("STEP 1: Loading and preprocessing sonar data...")
        X, y = detector.load_and_preprocess_data(csv_path)

        print("\nSTEP 2: Splitting data for training and validation...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        print("\nSTEP 3: Standardizing sonar features...")
        X_train_scaled, X_test_scaled = detector.prepare_features(X_train, X_test)

         # Train model
        print("\nSTEP 4: Training Random Forest classifier...")
        print("This may take a few minutes depending on your system...")
        detector.train_model(X_train_scaled, y_train, optimize_params=True)
        
        # Evaluate model
        print("\nSTEP 5: Evaluating model performance...")
        results = detector.evaluate_model(X_test_scaled, y_test, detailed=True)
        
        # Analyze feature importance
        print("\nSTEP 6: Analyzing frequency band importance...")
        feature_importance = detector.analyze_feature_importance()
        
        # Save the trained model
        print("\nSTEP 7: Saving trained model for deployment...")
        model_path = detector.save_model("operational_mine_detector")

        print("=" * 70)
        print("SYSTEM DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Your trained model is saved and ready for deployment.")
        print(f"Model files saved with base name: {model_path}")
        print("\nTo use this model in production:")
        print("1. Load the model using detector.load_model()")
        print("2. Call detector.predict_single_sample() with 60 sonar values")
        print("3. Get back 'M' or 'R' prediction with confidence score")

        return model_path
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find the data file '{csv_path}'")
        print("Please ensure your CSV file is in the correct location.")
        print("Expected format: 208 rows × 61 columns (60 features + 1 label)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Please check your data format and try again.")

if __name__ == "__main__":
    CSV_FILE_PATH = "sonarall_data.csv"

    model_path = model_training_implementation(CSV_FILE_PATH)