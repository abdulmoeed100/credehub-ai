import React, { useState } from 'react';

// ─── CONFIG ─────────────────────────────────────────────────
// Change this to your Render backend URL when deployed
const API_BASE = 'https://moeed77-credehub-ai.hf.space';
// const API_BASE = 'http://localhost:8000'; // for local dev

// ─── STYLES ─────────────────────────────────────────────────
const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  .assess-root {
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    background: linear-gradient(135deg, #e8f4f8 0%, #d4eee8 50%, #e8f0fe 100%);
    padding: 40px 20px;
    color: #0f2137;
  }

  /* ── SETUP SCREEN ── */
  .setup-card {
    max-width: 560px;
    margin: 60px auto;
    background: white;
    border-radius: 24px;
    padding: 48px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.08);
    text-align: center;
  }
  .setup-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #e8f8f5;
    color: #0d9488;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 100px;
    margin-bottom: 24px;
    letter-spacing: 0.5px;
  }
  .setup-title {
    font-size: 36px;
    font-weight: 800;
    color: #0f2137;
    margin: 0 0 12px;
    line-height: 1.2;
  }
  .setup-title span { color: #0d9488; }
  .setup-sub {
    color: #64748b;
    font-size: 15px;
    line-height: 1.6;
    margin: 0 0 36px;
  }
  .setup-label {
    text-align: left;
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 8px;
    display: block;
  }
  .setup-input {
    width: 100%;
    padding: 14px 18px;
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    font-size: 15px;
    font-family: inherit;
    color: #0f2137;
    outline: none;
    transition: border-color 0.2s;
    box-sizing: border-box;
    margin-bottom: 20px;
  }
  .setup-input:focus { border-color: #0d9488; }
  .count-options {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 32px;
  }
  .count-btn {
    padding: 14px 0;
    border-radius: 12px;
    border: 2px solid #e5e7eb;
    background: white;
    font-size: 16px;
    font-weight: 700;
    color: #374151;
    cursor: pointer;
    transition: all 0.2s;
  }
  .count-btn:hover { border-color: #0d9488; color: #0d9488; }
  .count-btn.active {
    background: #0d9488;
    border-color: #0d9488;
    color: white;
    box-shadow: 0 4px 16px rgba(13,148,136,0.3);
  }
  .start-btn {
    width: 100%;
    padding: 18px;
    background: #0f2137;
    color: white;
    border: none;
    border-radius: 14px;
    font-size: 16px;
    font-weight: 700;
    font-family: inherit;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
  }
  .start-btn:hover { background: #0d9488; transform: translateY(-1px); box-shadow: 0 8px 24px rgba(13,148,136,0.3); }
  .start-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
  .info-row {
    display: flex;
    gap: 12px;
    margin-top: 24px;
  }
  .info-chip {
    flex: 1;
    background: #f8fafc;
    border-radius: 10px;
    padding: 12px;
    font-size: 12px;
    color: #64748b;
    text-align: center;
    line-height: 1.5;
  }
  .info-chip strong { display: block; color: #0f2137; font-size: 14px; }

  /* ── QUIZ SCREEN ── */
  .quiz-wrap {
    max-width: 800px;
    margin: 0 auto;
  }
  .quiz-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
    flex-wrap: wrap;
    gap: 12px;
  }
  .progress-bar-wrap {
    flex: 1;
    min-width: 200px;
    background: rgba(255,255,255,0.6);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #0d9488, #14b8a6);
    border-radius: 100px;
    transition: width 0.4s ease;
  }
  .quiz-count-label {
    font-size: 14px;
    font-weight: 600;
    color: #0f2137;
    white-space: nowrap;
  }
  .unit-tag {
    background: #e8f8f5;
    color: #0d9488;
    font-size: 12px;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 100px;
    letter-spacing: 0.3px;
  }
  .diff-tag {
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 100px;
    letter-spacing: 0.3px;
  }
  .diff-easy   { background: #dcfce7; color: #16a34a; }
  .diff-medium { background: #fef9c3; color: #ca8a04; }
  .diff-hard   { background: #fee2e2; color: #dc2626; }

  .question-card {
    background: white;
    border-radius: 20px;
    padding: 36px 40px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    margin-bottom: 16px;
  }
  .q-number {
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 16px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .q-text {
    font-size: 20px;
    font-weight: 700;
    color: #0f2137;
    line-height: 1.5;
    margin-bottom: 28px;
  }
  .options-grid {
    display: grid;
    gap: 12px;
  }
  .option-btn {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 20px;
    border: 2px solid #e5e7eb;
    border-radius: 14px;
    background: white;
    cursor: pointer;
    text-align: left;
    font-family: inherit;
    font-size: 15px;
    color: #374151;
    transition: all 0.15s;
    width: 100%;
  }
  .option-btn:hover:not(:disabled) { border-color: #0d9488; background: #f0fdfa; }
  .option-btn.selected { border-color: #0d9488; background: #f0fdfa; color: #0f2137; }
  .option-btn.correct  { border-color: #16a34a; background: #dcfce7; color: #15803d; }
  .option-btn.wrong    { border-color: #dc2626; background: #fee2e2; color: #b91c1c; }
  .option-key {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: #f1f5f9;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 14px;
    flex-shrink: 0;
    transition: all 0.15s;
  }
  .selected .option-key { background: #0d9488; color: white; }
  .correct  .option-key { background: #16a34a; color: white; }
  .wrong    .option-key { background: #dc2626; color: white; }

  .explanation-box {
    margin-top: 16px;
    background: #f8fafc;
    border-left: 4px solid #0d9488;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    font-size: 14px;
    color: #475569;
    line-height: 1.6;
  }
  .explanation-box strong { color: #0f2137; display: block; margin-bottom: 4px; }

  .nav-btns {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-top: 8px;
  }
  .nav-btn {
    flex: 1;
    padding: 16px;
    border-radius: 14px;
    font-size: 15px;
    font-weight: 700;
    font-family: inherit;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
  }
  .nav-btn.secondary { background: #f1f5f9; color: #374151; }
  .nav-btn.secondary:hover { background: #e2e8f0; }
  .nav-btn.primary { background: #0f2137; color: white; }
  .nav-btn.primary:hover { background: #0d9488; box-shadow: 0 4px 16px rgba(13,148,136,0.3); }
  .nav-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .nav-btn.submit-btn { background: #0d9488; color: white; }
  .nav-btn.submit-btn:hover { background: #0f766e; }

  /* ── LOADING SCREEN ── */
  .loading-screen {
    min-height: 60vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 20px;
    text-align: center;
  }
  .spinner {
    width: 56px;
    height: 56px;
    border: 4px solid #e5e7eb;
    border-top-color: #0d9488;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-title { font-size: 20px; font-weight: 700; color: #0f2137; }
  .loading-sub { font-size: 14px; color: #64748b; max-width: 320px; line-height: 1.5; }
  .loading-steps { display: flex; gap: 6px; margin-top: 8px; }
  .loading-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #0d9488;
    animation: bounce 1.2s ease-in-out infinite;
  }
  .loading-dot:nth-child(2) { animation-delay: 0.2s; }
  .loading-dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-10px); } }

  /* ── REPORT SCREEN ── */
  .report-wrap { max-width: 860px; margin: 0 auto; }
  .report-hero {
    background: white;
    border-radius: 24px;
    padding: 40px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
  }
  .report-badge { display: inline-flex; align-items: center; gap: 8px; background: #e8f8f5; color: #0d9488; font-size: 13px; font-weight: 600; padding: 8px 16px; border-radius: 100px; margin-bottom: 20px; }
  .grade-circle {
    width: 100px; height: 100px;
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 36px; font-weight: 800;
    margin: 0 auto 20px;
  }
  .grade-Ap { background: linear-gradient(135deg,#d1fae5,#a7f3d0); color: #065f46; }
  .grade-A  { background: linear-gradient(135deg,#dbeafe,#bfdbfe); color: #1e3a8a; }
  .grade-B  { background: linear-gradient(135deg,#fef9c3,#fde68a); color: #78350f; }
  .grade-C  { background: linear-gradient(135deg,#fef3c7,#fcd34d); color: #92400e; }
  .grade-D  { background: linear-gradient(135deg,#fee2e2,#fecaca); color: #7f1d1d; }
  .grade-Fail { background: linear-gradient(135deg,#fecaca,#fca5a5); color: #450a0a; }
  .score-text { font-size: 42px; font-weight: 800; color: #0f2137; margin-bottom: 8px; }
  .score-text span { font-size: 22px; color: #64748b; font-weight: 500; }
  .student-name-label { font-size: 16px; color: #64748b; }
  .student-name-label strong { color: #0f2137; }

  .stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 24px 0 0;
  }
  .stat-chip { background: #f8fafc; border-radius: 14px; padding: 16px; text-align: center; }
  .stat-chip .val { font-size: 22px; font-weight: 800; color: #0f2137; }
  .stat-chip .lbl { font-size: 12px; color: #64748b; margin-top: 4px; }

  .section-title {
    font-size: 20px; font-weight: 800; color: #0f2137;
    margin: 28px 0 14px;
    display: flex; align-items: center; gap: 10px;
  }
  .unit-grid { display: grid; gap: 12px; }
  .unit-card {
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 16px;
  }
  .unit-name { font-size: 15px; font-weight: 600; color: #0f2137; margin-bottom: 8px; }
  .unit-bar-bg { background: #f1f5f9; border-radius: 100px; height: 6px; overflow: hidden; }
  .unit-bar { height: 100%; border-radius: 100px; transition: width 1s ease; }
  .bar-strong  { background: linear-gradient(90deg,#16a34a,#22c55e); }
  .bar-average { background: linear-gradient(90deg,#d97706,#f59e0b); }
  .bar-weak    { background: linear-gradient(90deg,#dc2626,#ef4444); }
  .unit-score { font-size: 18px; font-weight: 800; color: #0f2137; text-align: right; }
  .unit-status { font-size: 12px; font-weight: 600; margin-top: 2px; text-align: right; }
  .status-Strong  { color: #16a34a; }
  .status-Average { color: #d97706; }
  .status-Weak    { color: #dc2626; }

  .tags-row { display: flex; flex-wrap: wrap; gap: 8px; }
  .tag { padding: 8px 16px; border-radius: 100px; font-size: 13px; font-weight: 600; }
  .tag-strong  { background: #dcfce7; color: #15803d; }
  .tag-average { background: #fef9c3; color: #92400e; }
  .tag-weak    { background: #fee2e2; color: #b91c1c; }

  .narrative-card {
    background: white;
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    margin-top: 8px;
    line-height: 1.8;
    color: #374151;
    font-size: 15px;
    white-space: pre-wrap;
  }

  .retake-btn {
    width: 100%;
    margin-top: 24px;
    padding: 18px;
    background: #0f2137;
    color: white;
    border: none;
    border-radius: 14px;
    font-size: 16px;
    font-weight: 700;
    font-family: inherit;
    cursor: pointer;
    transition: all 0.2s;
  }
  .retake-btn:hover { background: #0d9488; transform: translateY(-1px); }

  @media (max-width: 600px) {
    .question-card { padding: 24px 20px; }
    .q-text { font-size: 17px; }
    .stat-row { grid-template-columns: 1fr 1fr; }
    .setup-card { padding: 32px 24px; }
    .count-options { grid-template-columns: repeat(2, 1fr); }
  }
`;

// ─── HELPER ─────────────────────────────────────────────────
function gradeClass(g) {
  if (g === 'A+') return 'grade-Ap';
  if (g === 'Fail') return 'grade-Fail';
  return `grade-${g}`;
}

// ─── MAIN COMPONENT ─────────────────────────────────────────
export default function Assessment() {
  const [screen, setScreen]         = useState('setup');   // setup | loading | quiz | submitting | report
  const [studentName, setStudentName] = useState('');
  const [numQuestions, setNumQuestions] = useState(20);
  const [questions, setQuestions]   = useState([]);
  const [current, setCurrent]       = useState(0);
  const [answers, setAnswers]       = useState({});       // { [id]: 'A'|'B'|'C'|'D' }
  const [showExplanation, setShowExplanation] = useState(false);
  const [report, setReport]         = useState(null);
  const [error, setError]           = useState('');

  const COUNT_OPTIONS = [10, 15, 20, 30];

  // ── START ASSESSMENT ──
  async function handleStart() {
    if (!studentName.trim()) { setError('Please enter your name.'); return; }
    setError('');
    setScreen('loading');
    try {
      const res = await fetch(`${API_BASE}/assessment/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_questions: numQuestions, subject: 'Computer Science', grade: 9 }),
      });
      if (!res.ok) throw new Error('Failed to generate questions.');
      const data = await res.json();
      setQuestions(data.questions);
      setAnswers({});
      setCurrent(0);
      setShowExplanation(false);
      setScreen('quiz');
    } catch (e) {
      setError('Could not connect to server. Please try again.');
      setScreen('setup');
    }
  }

  // ── SELECT ANSWER ──
  function handleSelect(key) {
    if (answers[questions[current].id]) return; // already answered
    const newAnswers = { ...answers, [questions[current].id]: key };
    setAnswers(newAnswers);
    setShowExplanation(true);
  }

  // ── NAVIGATE ──
  function goNext() {
    setShowExplanation(!!answers[questions[current + 1]?.id]);
    setCurrent(c => c + 1);
  }
  function goPrev() {
    setShowExplanation(!!answers[questions[current - 1]?.id]);
    setCurrent(c => c - 1);
  }

  // ── SUBMIT ──
  async function handleSubmit() {
    setScreen('submitting');
    const studentAnswers = Object.entries(answers).map(([qid, selected]) => ({
      question_id: parseInt(qid),
      selected,
    }));
    try {
      const res = await fetch(`${API_BASE}/assessment/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ questions, student_answers: studentAnswers, student_name: studentName }),
      });
      if (!res.ok) throw new Error('Failed to generate report.');
      const data = await res.json();
      setReport(data);
      setScreen('report');
    } catch (e) {
      setError('Could not generate report. Please try again.');
      setScreen('quiz');
    }
  }

  // ── RETAKE ──
  function handleRetake() {
    setReport(null);
    setQuestions([]);
    setAnswers({});
    setCurrent(0);
    setScreen('setup');
  }

  const q = questions[current];
  const answered = q ? !!answers[q.id] : false;
  const allAnswered = questions.length > 0 && questions.every(q => answers[q.id]);
  const progress = questions.length > 0 ? ((current + 1) / questions.length) * 100 : 0;

  return (
    <>
      <style>{styles}</style>
      <div className="assess-root">

        {/* ── SETUP SCREEN ── */}
        {screen === 'setup' && (
          <div className="setup-card">
            <div className="setup-badge">🎯 AI-Powered Assessment</div>
            <h1 className="setup-title">
              Test Your <span>Preparation</span>
            </h1>
            <p className="setup-sub">
              AI generates fresh MCQs from all 7 chapters of your Class 9 CS book
              — just like Karachi Board exam style.
            </p>

            <label className="setup-label">Your Name</label>
            <input
              className="setup-input"
              placeholder="Enter your name..."
              value={studentName}
              onChange={e => setStudentName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleStart()}
            />

            <label className="setup-label">Number of Questions</label>
            <div className="count-options">
              {COUNT_OPTIONS.map(n => (
                <button
                  key={n}
                  className={`count-btn ${numQuestions === n ? 'active' : ''}`}
                  onClick={() => setNumQuestions(n)}
                >{n}</button>
              ))}
            </div>

            {error && <p style={{ color: '#dc2626', marginBottom: 16, fontSize: 14 }}>{error}</p>}

            <button className="start-btn" onClick={handleStart}>
              ⚡ Start Assessment
            </button>

            <div className="info-row">
              <div className="info-chip"><strong>📚 All Chapters</strong>Questions from every unit</div>
              <div className="info-chip"><strong>🤖 AI Generated</strong>Fresh MCQs every time</div>
              <div className="info-chip"><strong>📊 Full Report</strong>Weak topics identified</div>
            </div>
          </div>
        )}

        {/* ── LOADING SCREEN (Generating MCQs) ── */}
        {(screen === 'loading' || screen === 'submitting') && (
          <div className="loading-screen">
            <div className="spinner" />
            <div className="loading-title">
              {screen === 'loading' ? 'Generating Your Questions...' : 'Analyzing Your Performance...'}
            </div>
            <p className="loading-sub">
              {screen === 'loading'
                ? `AI is creating ${numQuestions} unique MCQs from all 7 chapters of your CS book.`
                : 'AI is preparing your personalized assessment report with unit-wise analysis.'}
            </p>
            <div className="loading-steps">
              <div className="loading-dot" />
              <div className="loading-dot" />
              <div className="loading-dot" />
            </div>
          </div>
        )}

        {/* ── QUIZ SCREEN ── */}
        {screen === 'quiz' && q && (
          <div className="quiz-wrap">
            {/* Header */}
            <div className="quiz-header">
              <span className="quiz-count-label">Question {current + 1} of {questions.length}</span>
              <div className="progress-bar-wrap">
                <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <span className="unit-tag">{q.unit.split(' - ')[1] || q.unit}</span>
                <span className={`diff-tag diff-${q.difficulty}`}>{q.difficulty}</span>
              </div>
            </div>

            {/* Question Card */}
            <div className="question-card">
              <div className="q-number">Q{current + 1} · {q.topic}</div>
              <div className="q-text">{q.question}</div>

              <div className="options-grid">
                {['A', 'B', 'C', 'D'].map(key => {
                  const isSelected = answers[q.id] === key;
                  const isCorrect  = answered && key === q.correct_answer;
                  const isWrong    = answered && isSelected && key !== q.correct_answer;
                  return (
                    <button
                      key={key}
                      className={`option-btn ${isCorrect ? 'correct' : ''} ${isWrong ? 'wrong' : ''} ${isSelected && !isWrong && !isCorrect ? 'selected' : ''}`}
                      onClick={() => handleSelect(key)}
                      disabled={answered}
                    >
                      <span className="option-key">{key}</span>
                      {q.options[key]}
                    </button>
                  );
                })}
              </div>

              {showExplanation && answered && (
                <div className="explanation-box">
                  <strong>💡 Explanation</strong>
                  {q.explanation}
                </div>
              )}
            </div>

            {/* Navigation */}
            <div className="nav-btns">
              <button className="nav-btn secondary" onClick={goPrev} disabled={current === 0}>← Previous</button>
              {current < questions.length - 1 ? (
                <button className="nav-btn primary" onClick={goNext}>Next →</button>
              ) : (
                <button
                  className="nav-btn submit-btn"
                  onClick={handleSubmit}
                  disabled={!allAnswered}
                  title={!allAnswered ? 'Answer all questions to submit' : ''}
                >
                  {allAnswered ? '📊 Submit & Get Report' : `Answer all questions (${Object.keys(answers).length}/${questions.length})`}
                </button>
              )}
            </div>
          </div>
        )}

        {/* ── REPORT SCREEN ── */}
        {screen === 'report' && report && (
          <div className="report-wrap">

            {/* Hero Score Card */}
            <div className="report-hero">
              <div className="report-badge">📊 Assessment Complete</div>
              <div className={`grade-circle ${gradeClass(report.grade_label)}`}>
                {report.grade_label}
              </div>
              <div className="score-text">
                {report.total_score}<span>/{report.total_questions}</span>
              </div>
              <div className="student-name-label">
                <strong>{report.student_name}</strong> · Class 9 Computer Science
              </div>
              <div className="stat-row">
                <div className="stat-chip">
                  <div className="val">{report.percentage}%</div>
                  <div className="lbl">Overall Score</div>
                </div>
                <div className="stat-chip">
                  <div className="val" style={{ color: '#16a34a' }}>{report.strong_units.length}</div>
                  <div className="lbl">Strong Units</div>
                </div>
                <div className="stat-chip">
                  <div className="val" style={{ color: '#dc2626' }}>{report.weak_units.length}</div>
                  <div className="lbl">Weak Units</div>
                </div>
              </div>
            </div>

            {/* Unit Breakdown */}
            <div className="section-title">📚 Unit-wise Performance</div>
            <div className="unit-grid">
              {report.unit_results.map(r => (
                <div className="unit-card" key={r.unit}>
                  <div style={{ flex: 1 }}>
                    <div className="unit-name">{r.unit}</div>
                    <div className="unit-bar-bg">
                      <div
                        className={`unit-bar bar-${r.status.toLowerCase()}`}
                        style={{ width: `${r.score_pct}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="unit-score">{r.score_pct}%</div>
                    <div className={`unit-status status-${r.status}`}>{r.status}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Strong / Weak Tags */}
            {report.strong_units.length > 0 && (
              <>
                <div className="section-title">✅ Strong Areas</div>
                <div className="tags-row">
                  {report.strong_units.map(u => (
                    <span key={u} className="tag tag-strong">{u.split(' - ')[1] || u}</span>
                  ))}
                </div>
              </>
            )}
            {report.average_units.length > 0 && (
              <>
                <div className="section-title">⚠️ Average — Needs Work</div>
                <div className="tags-row">
                  {report.average_units.map(u => (
                    <span key={u} className="tag tag-average">{u.split(' - ')[1] || u}</span>
                  ))}
                </div>
              </>
            )}
            {report.weak_units.length > 0 && (
              <>
                <div className="section-title">❌ Weak Areas — Focus Here</div>
                <div className="tags-row">
                  {report.weak_units.map(u => (
                    <span key={u} className="tag tag-weak">{u.split(' - ')[1] || u}</span>
                  ))}
                </div>
              </>
            )}

            {/* AI Narrative */}
            <div className="section-title">🤖 AI Assessment Report</div>
            <div className="narrative-card">{report.ai_narrative}</div>

            {/* Retake */}
            <button className="retake-btn" onClick={handleRetake}>🔄 Take Another Assessment</button>
          </div>
        )}
      </div>
    </>
  );
}
