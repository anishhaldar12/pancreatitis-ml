import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  FEATURE_KEYS,
  createModel,
  trainModel,
  predictProb,
  normalizeVector,
  saveLocal,
  loadLocalIfAny
} from './ml/model';
import { demoSamples } from './ml/sampleData';

const tests = [
  { name: 'Amylase', min: 30, max: 110, unit: 'U/L', description: 'Pancreatic enzyme; elevated in acute pancreatitis.' },
  { name: 'Lipase', min: 0, max: 160, unit: 'U/L', description: 'Specific pancreatic enzyme; high levels indicate pancreatitis.' },
  { name: 'Calcium (Ca²⁺)', min: 8.5, max: 10.2, unit: 'mg/dL', description: 'Important for bone & heart health; low in severe pancreatitis.' },
  { name: 'Potassium (K⁺)', min: 3.5, max: 5.0, unit: 'mmol/L', description: 'Electrolyte; low levels affect heart rhythm.' },
  { name: 'Magnesium (Mg²⁺)', min: 1.7, max: 2.2, unit: 'mg/dL', description: 'Electrolyte; deficiency may occur in pancreatitis.' },
  { name: 'C-Reactive Protein (CRP)', min: 0, max: 10, unit: 'mg/L', description: 'Inflammation marker; high in acute pancreatitis.' },
  { name: 'White Blood Cell (WBC)', min: 4000, max: 11000, unit: '/μL', description: 'Elevated in infection or inflammation.' },
  { name: 'ALT (SGPT)', min: 7, max: 56, unit: 'U/L', description: 'Liver enzyme; elevated in liver involvement.' },
  { name: 'AST (SGOT)', min: 10, max: 40, unit: 'U/L', description: 'Liver enzyme; check with ALT.' },
  { name: 'Bilirubin (Total)', min: 0.3, max: 1.2, unit: 'mg/dL', description: 'High levels indicate liver or bile duct issues.' },
  { name: 'Albumin', min: 3.4, max: 5.4, unit: 'g/dL', description: 'Protein level; low may indicate malnutrition.' },
  { name: 'Vitamin D', min: 30, max: 100, unit: 'ng/mL', description: 'Low levels affect immunity and bone health.' },
  { name: 'Fasting Blood Sugar', min: 70, max: 99, unit: 'mg/dL', description: 'High levels indicate impaired glucose regulation.' }
];

export default function App() {
  const [values, setValues] = useState({});
  const [finalResult, setFinalResult] = useState('');
  const [model, setModel] = useState(null);
  const [mlProb, setMlProb] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainStatus, setTrainStatus] = useState('');

  useEffect(() => {
    (async () => {
      const m = await loadLocalIfAny();
      if (m) setModel(m);
    })();
  }, []);

  const handleChange = (e, test) => {
    setValues({ ...values, [test.name]: parseFloat(e.target.value) });
  };

  const getResultColor = (test) => {
    const value = values[test.name];
    if (value == null) return 'border-gray-300';
    if (value < test.min) return 'border-amber-500';
    if (value > test.max) return 'border-red-600';
    return 'border-green-600';
  };

  const ruleCheck = () => {
    let findings = [];
    if (values['Amylase'] > 110) findings.push('Amylase High');
    if (values['Lipase'] > 160) findings.push('Lipase High (specific)');
    if (values['Calcium (Ca²⁺)'] < 8.5) findings.push('Low Calcium');
    if (values['Potassium (K⁺)'] < 3.5) findings.push('Low Potassium');
    if (values['Magnesium (Mg²⁺)'] < 1.7) findings.push('Low Magnesium');
    if (values['C-Reactive Protein (CRP)'] > 10) findings.push('High CRP');
    if (values['White Blood Cell (WBC)'] > 11000) findings.push('High WBC');
    if (values['ALT (SGPT)'] > 56) findings.push('High ALT');
    if (values['AST (SGOT)'] > 40) findings.push('High AST');
    if (values['Bilirubin (Total)'] > 1.2) findings.push('High Bilirubin');
    if (values['Albumin'] < 3.4) findings.push('Low Albumin');
    if (values['Vitamin D'] < 30) findings.push('Low Vitamin D');
    if (values['Fasting Blood Sugar'] > 99) findings.push('High Blood Sugar');

    if ((values['Lipase'] > 160 || values['Amylase'] > 110) && findings.length > 2) {
      setFinalResult('⚠️ Possible Pancreatitis detected. Please consult a doctor.');
    } else if (findings.length > 0) {
      setFinalResult('⚠️ Some abnormalities present. Consider medical advice.');
    } else {
      setFinalResult('✅ No significant signs of Pancreatitis.');
    }
  };

  const trainDemo = async () => {
    setTraining(true);
    setTrainStatus('Preparing data...');
    const xsNorm = demoSamples.map(s => {
      const row = {};
      FEATURE_KEYS.forEach(k => row[k] = s.x[k]);
      return normalizeVector(row);
    });
    const ys = demoSamples.map(s => s.y);
    setTrainStatus('Creating model...');
    const m = createModel();
    setTrainStatus('Training...');
    await trainModel(m, xsNorm, ys, {
      onEpochEnd: (epoch, logs) => {
        setTrainStatus(`Epoch ${epoch + 1} | loss: ${logs.loss.toFixed(4)} | acc: ${(logs.acc ?? logs.accuracy ?? 0).toFixed(3)}`);
      }
    });
    setTrainStatus('Saving model...');
    await saveLocal(m);
    setModel(m);
    setTraining(false);
    setTrainStatus('✅ Trained & Saved locally.');
  };

  const predict = async () => {
    if (!model) { setTrainStatus('No model loaded. Train demo model first.'); return; }
    const raw = {};
    FEATURE_KEYS.forEach(k => raw[k] = values[k] ?? 0);
    const p = await predictProb(model, raw);
    setMlProb(p);
  };

  return (
    <div className='flex flex-col items-center justify-center min-h-screen p-6 bg-gray-100'>
      <motion.div className='app-card bg-white rounded-xl shadow-lg p-8 w-full max-w-5xl' initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
        <h1 className='text-3xl font-bold text-indigo-700 text-center mb-8'>🧪 Pancreatitis Lab Checker + ML</h1>

        <div className='grid md:grid-cols-2 gap-6'>
          {tests.map(test => (
            <div key={test.name} className='flex flex-col relative'>
              <label className='label-small'>{test.name}</label>
              <input
                className={`input-field border-2 ${getResultColor(test)}`}
                type='number'
                placeholder={`${test.min} - ${test.max} ${test.unit}`}
                onChange={(e) => handleChange(e, test)}
              />
              <span className='absolute right-2 top-1/2 transform -translate-y-1/2 text-xs text-gray-500' title={test.description}>
                ℹ️
              </span>
            </div>
          ))}
        </div>

        <div className='grid md:grid-cols-3 gap-4 mt-8'>
          <button onClick={ruleCheck} className='px-6 py-3 rounded-lg bg-indigo-600 text-white font-semibold shadow hover:bg-indigo-700 transition'>Rule-based Check</button>
          <button onClick={trainDemo} disabled={training} className='px-6 py-3 rounded-lg bg-emerald-600 text-white font-semibold shadow hover:bg-emerald-700 transition'>{training ? 'Training...' : 'Train Demo Model'}</button>
          <button onClick={predict} className='px-6 py-3 rounded-lg bg-fuchsia-600 text-white font-semibold shadow hover:bg-fuchsia-700 transition'>Predict with ML</button>
        </div>

        {trainStatus && <div className='mt-4 text-sm text-gray-600'>{trainStatus}</div>}

        {finalResult && (
          <motion.div className={`result-box mt-4 ${finalResult.includes('⚠️') ? 'result-bad' : 'result-good'}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            {finalResult}
          </motion.div>
        )}

        {mlProb != null && (
          <motion.div className='mt-4 p-4 rounded-lg text-lg font-semibold bg-blue-50 text-blue-800' initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            ML Risk Score: {(mlProb * 100).toFixed(1)}%
          </motion.div>
        )}

        <p className='mt-6 text-xs text-gray-500 text-center'>Educational use only. Not a medical device. Always consult a clinician for diagnosis.</p>
      </motion.div>
    </div>
  );
}
