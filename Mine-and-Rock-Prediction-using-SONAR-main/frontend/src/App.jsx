import { useState, useEffect, useRef } from 'react'
import { Zap, AlertTriangle, Shield, Play, Pause, RotateCcw } from 'lucide-react'
import './App.css'

function App() {
  const [manualInput, setManualInput] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [simulation, setSimulation] = useState({
    running: false,
    currentIndex: 0,
    results: [],
    paused: false,
    acknowledged: true
  })
  const [error, setError] = useState('')

  const audioContextRef = useRef(null)
  const isSimulationRunningRef = useRef(false)
  const simulationIndexRef = useRef(0)
  const beepIntervalRef = useRef(null);
  
  useEffect(() => {
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }

      if (beepIntervalRef.current) {
        clearInterval(beepIntervalRef.current);
      }
    };
  }, []);

  // Beep sound function
  const playBeep = (frequency = 800, duration = 500) => {
    if (!audioContextRef.current) return;
    
    const oscillator = audioContextRef.current.createOscillator();
    const gainNode = audioContextRef.current.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContextRef.current.destination);
    
    oscillator.frequency.value = frequency;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContextRef.current.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContextRef.current.currentTime + duration / 1000);
    
    oscillator.start(audioContextRef.current.currentTime);
    oscillator.stop(audioContextRef.current.currentTime + duration / 1000);
  }
  
  const startContinuousBeep = () => {
    if (beepIntervalRef.current) {
      clearInterval(beepIntervalRef.current);
    }

    playBeep(1000, 600); // Initial beep
    beepIntervalRef.current = setInterval(() => {
      playBeep(1000, 600);
    }, 1000); // Beep every 2 seconds
  }
  
  const stopContinuousBeep = () => {
    if (beepIntervalRef.current) {
      clearInterval(beepIntervalRef.current);
      beepIntervalRef.current = null;
    }
  }

  const handleManualPredict = async () => {
    if (!manualInput.trim()) {
      setError('Please enter sonar values.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: manualInput })
      })

      if (!response.ok) {
        throw new Error('Failed to fetch prediction')
      }

      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const startSimulation = async () => {
    try {
      const response = await fetch('http://localhost:5000/simulation/start', {
        method: 'POST',
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setSimulation(prev => ({
          ...prev,
          results: data.results,
          running: true,
          currentIndex: 0,
          acknowledged: true
        }));
        isSimulationRunningRef.current = true;
        runSimulation(data.results);
      } else {
        setError(data.error || 'Simulation failed to start');
      }
    } catch (err) {
      setError('Failed to start simulation');
    }
  };

  // const runSimulation = (results) => {
  //   let index = 0;
  //   const processNextItem = () => {
  //     if (index >= results.length) {
  //       stopSimulation();
  //       return;
  //     }

  //     const currentResult = results[index];
      
  //     setSimulation(prev => ({
  //       ...prev,
  //       currentIndex: index
  //     }));

  //     // If mine detected, beep and wait for acknowledgment
  //     if (currentResult.prediction === 'M') {
  //       playBeep(1000, 600);
  //       setSimulation(prev => ({
  //         ...prev,
  //         acknowledged: false,
  //         paused: true
  //       }));
  //       // Don't increment index until acknowledged
  //       return;
  //     }

  //     // If rock, continue automatically
  //     index++;
  //     setTimeout(processNextItem, 2000); // 2 second intervals
  //   }

  //   processNextItem();
  // }

  // const acknowledgeAlert = () => {
  //   setSimulation(prev => ({
  //     ...prev,
  //     acknowledged: true,
  //     paused: false,
  //   }));

  //   // Continue simulation from next index
  //   setTimeout(() => {
  //     setSimulation(prev => {
  //       const nextIndex = prev.currentIndex + 1;
  //       if (nextIndex < prev.results.length) {
  //         setSimulation(current => ({
  //           ...current,
  //           currentIndex: nextIndex
  //         }));

  //         const continueSimulation = () => {
  //           let currentIndex = nextIndex;
  //           const processNextItem = () => {
  //             if (currentIndex >= prev.results.length) {
  //               stopSimulation();
  //               return
  //             }

  //             const currentResult = prev.results[currentIndex]

  //             setSimulation(current => ({
  //               ...current,
  //               currentIndex: currentIndex
  //             }))

  //             if (currentResult.prediction === 'M') {
  //               playBeep(1000, 600);
  //               setSimulation(current => ({
  //                 ...current,
  //                 acknowledged: false,
  //                 paused: true
  //               }));
  //               return;
  //             }
  //             currentIndex++;
  //             setTimeout(processNextItem, 2000); // 2 second intervals

  //           }
  //           processNextItem();
  //         }
  //         setTimeout(continueSimulation, 1000); // Delay before continuing
  //         return prev;
  //       } else {
  //         stopSimulation();
  //         return prev;
  //       }
  //     });
  //   }, 500);
  // }

  const runSimulation = (results) => {
    simulationIndexRef.current = 0;
    processSimulationStep(results);
  }

  const processSimulationStep = (results) => {
    if(!isSimulationRunningRef.current) return;
    const currentIndex = simulationIndexRef.current;
    if (currentIndex >= results.length) {
      stopSimulation();
      return;
    }
    console.log('running')
    const currentResult = results[currentIndex];
    setSimulation(prev => ({
      ...prev,
      currentIndex: currentIndex
    }));
    if (currentResult.prediction === 'M') {
      startContinuousBeep();
      setSimulation(prev => ({
        ...prev,
        acknowledged: false,
        paused: true
      }));

      return;
    }

    simulationIndexRef.current++;
    setTimeout(() => processSimulationStep(results), 2000); // 2 second intervals
    // processSimulationStep(results); // Continue immediately for rocks
  }

  const acknowledgeAlert = () => {
    stopContinuousBeep();
    setSimulation(prev => ({
      ...prev,
      acknowledged: true,
      paused: false,
    }));

    simulationIndexRef.current++;

    setTimeout(() => {
      if (simulationIndexRef.current < simulation.results.length) {
        processSimulationStep(simulation.results);
      }
      else {
        stopSimulation();
      }
    }, 1000); // Delay before continuing
  }

  const stopSimulation = () => {
    stopContinuousBeep()
    isSimulationRunningRef.current = false;
    setSimulation(prev => ({
      ...prev,
      running: false,
      paused: false
    }));
  };

  const resetSimulation = () => {
    stopSimulation();
    setSimulation({
      running: false,
      currentIndex: 0,
      results: [],
      paused: false,
      acknowledged: true
    });
  };

  const getCurrentResult = () => {
    if (simulation.results.length > 0 && simulation.currentIndex < simulation.results.length) {
      return simulation.results[simulation.currentIndex];
    }
    return null;
  };

  const currentResult = getCurrentResult();

  return (
    <>
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-white mb-8 text-center flex items-center justify-center gap-3">
          <Zap className="text-yellow-400" />
          Sonar Detection System
        </h1>

        {/* Manual Prediction Section */}
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 mb-8 border border-white/20">
          <h2 className="text-2xl font-semibold text-white mb-4">Manual Prediction</h2>
          
          <div className="space-y-4">
            <textarea
              value={manualInput}
              onChange={(e) => setManualInput(e.target.value)}
              placeholder="Enter 60 sonar values separated by commas..."
              className="w-full h-27 p-3 rounded-lg bg-white/20 text-white placeholder-white/70 border border-white/30 focus:border-blue-400 focus:outline-none resize-none"
            />
            
            <button
              onClick={handleManualPredict}
              disabled={loading}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-lg font-medium transition-colors"
            >
              {loading ? 'Analyzing...' : 'Predict'}
            </button>
          </div>

          {prediction && (
            <div className={`mt-4 p-4 rounded-lg ${
              prediction.prediction === 'M' 
                ? 'bg-red-500/20 border border-red-500/50' 
                : 'bg-green-500/20 border border-green-500/50'
            }`}>
              <div className="flex items-center gap-2 mb-2">
                {prediction.prediction === 'M' ? (
                  <AlertTriangle className="text-red-400" size={24} />
                ) : (
                  <Shield className="text-green-400" size={24} />
                )}
                <span className="text-white font-bold text-lg">
                  Prediction: {prediction.prediction === 'M' ? 'Mine Detected' : 'Rock Detected'}
                </span>
              </div>
              <div className="text-white/80">
                <p>Mine Confidence: {(prediction.probabilities[0] * 100).toFixed(1)}%</p>
                <p>Rock Confidence: {(prediction.probabilities[1] * 100).toFixed(1)}%</p>
              </div>
            </div>
          )}
        </div>

        {/* Simulation Section */}
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
          <h2 className="text-2xl font-semibold text-white mb-4">Automated Simulation</h2>
          
          <div className="flex gap-4 mb-6">
            <button
              onClick={startSimulation}
              disabled={simulation.running}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white rounded-lg font-medium transition-colors"
            >
              <Play size={16} />
              Start Simulation
            </button>
            
            <button
              onClick={stopSimulation}
              disabled={!simulation.running}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-400 text-white rounded-lg font-medium transition-colors"
            >
              <Pause size={16} />
              Stop
            </button>
            
            <button
              onClick={resetSimulation}
              className="flex items-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
            >
              <RotateCcw size={16} />
              Reset
            </button>
          </div>

          {/* Current Detection Display */}
          {simulation.running && currentResult && (
            <div className={`p-6 rounded-xl border-2 transition-all duration-500 ${
              currentResult.prediction === 'M' && !simulation.acknowledged
                ? 'bg-red-500/30 border-red-500 animate-pulse'
                : currentResult.prediction === 'M'
                ? 'bg-red-500/20 border-red-500/50'
                : 'bg-green-500/20 border-green-500/50'
            }`}>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  {currentResult.prediction === 'M' ? (
                    <AlertTriangle className="text-red-400" size={32} />
                  ) : (
                    <Shield className="text-green-400" size={32} />
                  )}
                  <div>
                    <h3 className="text-2xl font-bold text-white">
                      Detection #{simulation.currentIndex + 1}
                    </h3>
                    <p className="text-white/80">
                      Status: {currentResult.prediction}
                    </p>
                  </div>
                </div>
                
                <div className="text-right text-white/80">
                  <p>Mine: {(currentResult.probabilities[0] * 100).toFixed(1)}%</p>
                  <p>Rock: {(currentResult.probabilities[1] * 100).toFixed(1)}%</p>
                </div>
              </div>

              {currentResult.prediction === 'M' && !simulation.acknowledged && (
                <div className="text-center">
                  <p className="text-white font-bold mb-4 text-lg">
                    ⚠️ MINE DETECTED ⚠️
                  </p>
                  <button
                    onClick={acknowledgeAlert}
                    className="px-8 py-3 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg font-bold text-lg transition-colors"
                  >
                    ACKNOWLEDGE
                  </button>
                </div>
              )}

              {/* Progress indicator */}
              {/* <div className="mt-4">
                <div className="flex justify-between text-white/70 text-sm mb-1">
                  <span>Progress</span>
                  <span>{simulation.currentIndex + 1} / {simulation.results.length}</span>
                </div>
                <div className="w-full bg-white/20 rounded-full h-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${((simulation.currentIndex + 1) / simulation.results.length) * 100}%` }}
                  />
                </div>
              </div> */}
            </div>
          )}

          {/* Simulation Status */}
          {simulation.running && (
            <div className="mt-4 text-center">
              <p className="text-white/80">
                {simulation.paused ? 'Simulation Paused - Waiting for acknowledgment...' : 'Simulation Running...'}
              </p>
            </div>
          )}
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
            <p className="text-red-400">{error}</p>
          </div>
        )}
      </div>
      <footer className="text-center text-white/70 text-sm mt-8">
        <p>© 2023 Sonar Detection System. All rights reserved.</p>
        <p>Developed by Mohammed Ayaan</p>

      </footer>
      </div>
    </>
  )
}

export default App
