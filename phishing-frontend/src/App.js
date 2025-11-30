const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
import React, { useState } from 'react';
import { Shield, ShieldAlert, CheckCircle, Mail, Globe, Play, RotateCcw, Terminal, Zap } from 'lucide-react';

// --- Tab Component ---
const TabButton = ({ active, onClick, icon: Icon, label }) => (
  <button
    onClick={onClick}
    className={`flex-1 py-3 px-4 flex items-center justify-center gap-2 text-sm font-medium transition-all border-b-2 ${
      active 
        ? 'border-blue-500 text-blue-400 bg-slate-800/50' 
        : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/30'
    }`}
  >
    <Icon className="w-4 h-4" />
    {label}
  </button>
);

// --- Feature Definitions ---
const initialWebsiteFeatures = {
  having_ip_address: 0, url_length: 0, shortining_service: 0, having_at_symbol: 0,
  double_slash_redirecting: 0, prefix_suffix: 0, having_sub_domain: 0, sslfinal_state: 0,
  domain_registration_length: 0, favicon: 0, port: 0, https_token: 0, request_url: 0,
  url_of_anchor: 0, links_in_tags: 0, sfh: 0, submitting_to_email: 0, abnormal_url: 0,
  redirect: 0, on_mouseover: 0, rightclick: 0, popupwindow: 0, iframe: 0,
  age_of_domain: 0, dnsrecord: 0, web_traffic: 0, page_rank: 0, google_index: 0,
  links_pointing_to_page: 0, statistical_report: 0
};

// --- Presets for Quick Testing ---
const PRESETS = {
  phishing: {
    ...initialWebsiteFeatures,
    having_ip_address: 1,
    url_length: -1,
    sslfinal_state: -1,
    having_sub_domain: -1,
    web_traffic: -1,
    domain_registration_length: -1
  },
  safe: {
    ...initialWebsiteFeatures,
    having_ip_address: -1,
    sslfinal_state: 1,
    url_of_anchor: 1,
    having_sub_domain: 1,
    web_traffic: 1
  }
};

export default function App() {
  const [mode, setMode] = useState('email'); // 'email' or 'website'
  const [emailText, setEmailText] = useState('');
  const [features, setFeatures] = useState(initialWebsiteFeatures);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- Handlers ---
  const loadPreset = (type) => {
    setFeatures(PRESETS[type]);
    setResult(null);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    const payload = mode === 'email' 
      ? { email_text: emailText } 
      : { website_features: features };

    try {
      const response = await fetch('${API_BASE_URL}/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to connect to backend. Is uvicorn running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-500/30">
      
      {/* Header */}
      <header className="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center justify-between sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg shadow-lg shadow-blue-900/20">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">Sentin<span className="text-blue-500">AI</span> Hybrid</h1>
            <p className="text-xs text-slate-400 uppercase tracking-wider">Phishing Detection System</p>
          </div>
        </div>
        <div className="px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs font-mono text-slate-400 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
          API v2.0 Ready
        </div>
      </header>

      <main className="container mx-auto max-w-5xl p-6 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* LEFT PANEL: INPUT */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
            <div className="flex border-b border-slate-800">
              <TabButton active={mode === 'email'} onClick={() => setMode('email')} icon={Mail} label="Email Text Analysis" />
              <TabButton active={mode === 'website'} onClick={() => setMode('website')} icon={Globe} label="Website Feature Analysis" />
            </div>

            <div className="p-6">
              {mode === 'email' ? (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <label className="text-sm font-medium text-slate-300">Paste Email Content</label>
                    <button onClick={() => setEmailText("URGENT: Your account has been suspended. Click here to verify: http://bit.ly/secure-login")} className="text-xs text-blue-400 hover:text-blue-300 hover:underline">
                      Load Phishing Example
                    </button>
                  </div>
                  <textarea
                    value={emailText}
                    onChange={(e) => setEmailText(e.target.value)}
                    placeholder="Paste raw email text here..."
                    className="w-full h-64 bg-slate-950 border border-slate-800 rounded-lg p-4 text-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none resize-none font-mono text-sm leading-relaxed"
                  />
                </div>
              ) : (
                <div className="space-y-4">
                  {/* --- NEW PRESET BUTTONS --- */}
                  <div className="flex items-center justify-between mb-4 bg-slate-800/30 p-2 rounded-lg border border-slate-800">
                    <div className="flex gap-2">
                       <button onClick={() => loadPreset('phishing')} className="flex items-center gap-1 px-2 py-1 text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20 rounded hover:bg-red-500/20 transition-colors">
                         <Zap className="w-3 h-3" /> Load Phishing
                       </button>
                       <button onClick={() => loadPreset('safe')} className="flex items-center gap-1 px-2 py-1 text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded hover:bg-emerald-500/20 transition-colors">
                         <Shield className="w-3 h-3" /> Load Safe
                       </button>
                    </div>
                    <button onClick={() => setFeatures(initialWebsiteFeatures)} className="text-xs text-slate-500 hover:text-white px-2">Reset</button>
                  </div>
                  {/* --- END NEW BUTTONS --- */}

                  <div className="grid grid-cols-2 gap-2 h-64 overflow-y-auto pr-2 custom-scrollbar">
                    {Object.keys(features).map(key => (
                      <div key={key} className="bg-slate-950 p-2 rounded border border-slate-800 flex items-center justify-between">
                        <span className="text-[10px] uppercase text-slate-500 truncate w-24" title={key}>{key.replace(/_/g, ' ')}</span>
                        <select 
                          value={features[key]}
                          onChange={(e) => setFeatures({...features, [key]: parseInt(e.target.value)})}
                          className="bg-slate-800 text-xs rounded p-1 border-none focus:ring-1 focus:ring-blue-500"
                        >
                          <option value="1">1</option>
                          <option value="0">0</option>
                          <option value="-1">-1</option>
                        </select>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            
            <div className="p-4 bg-slate-800/50 border-t border-slate-800 flex justify-end">
              <button
                onClick={handlePredict}
                disabled={loading}
                className={`py-2 px-6 rounded-lg font-semibold text-sm flex items-center gap-2 transition-all ${
                  loading 
                    ? 'bg-slate-700 text-slate-400 cursor-not-allowed' 
                    : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20 hover:scale-105 active:scale-95'
                }`}
              >
                {loading ? <RotateCcw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                {loading ? 'Analyzing...' : 'Analyze Threat'}
              </button>
            </div>
          </div>
        </div>

        {/* RIGHT PANEL: RESULT (No changes needed here) */}
        <div className="lg:col-span-5 space-y-6">
          {result ? (
            <div className={`h-full min-h-[300px] rounded-xl p-8 flex flex-col items-center justify-center text-center shadow-2xl animate-in fade-in slide-in-from-bottom-4 ${
              result.prediction === "Phishing" 
                ? 'bg-gradient-to-b from-red-500/10 to-slate-900 border border-red-500/30' 
                : 'bg-gradient-to-b from-emerald-500/10 to-slate-900 border border-emerald-500/30'
            }`}>
              
              {result.prediction === "Phishing" ? (
                <div className="w-24 h-24 bg-red-500/20 rounded-full flex items-center justify-center mb-6 shadow-[0_0_30px_rgba(239,68,68,0.3)]">
                  <ShieldAlert className="w-12 h-12 text-red-500" />
                </div>
              ) : (
                <div className="w-24 h-24 bg-emerald-500/20 rounded-full flex items-center justify-center mb-6 shadow-[0_0_30px_rgba(16,185,129,0.3)]">
                  <CheckCircle className="w-12 h-12 text-emerald-500" />
                </div>
              )}

              <h2 className={`text-4xl font-black tracking-tighter mb-2 ${
                result.prediction === "Phishing" ? 'text-red-400' : 'text-emerald-400'
              }`}>
                {result.prediction.toUpperCase()}
              </h2>
              
              <div className="flex items-center gap-2 mb-8">
                <span className="text-slate-400 text-sm">Confidence:</span>
                <span className="text-white font-mono font-bold text-lg">{(parseFloat(result.confidence) * 100).toFixed(1)}%</span>
              </div>

              <div className="w-full bg-slate-900/50 rounded-lg p-4 border border-slate-800">
                <div className="grid grid-cols-2 gap-4 text-center divide-x divide-slate-800">
                  <div>
                    <div className="text-[10px] uppercase text-slate-500 tracking-wider mb-1">Input Type</div>
                    <div className="text-sm text-blue-300 font-medium">{result.type}</div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase text-slate-500 tracking-wider mb-1">Scan ID</div>
                    <div className="text-sm text-slate-300 font-mono">{Math.random().toString(36).substr(2, 6).toUpperCase()}</div>
                  </div>
                </div>
              </div>

            </div>
          ) : (
            <div className="h-full min-h-[300px] bg-slate-900/50 border border-slate-800 border-dashed rounded-xl flex flex-col items-center justify-center text-slate-600 p-8 text-center">
              <div className="bg-slate-900 p-4 rounded-full mb-4">
                <Terminal className="w-8 h-8 text-slate-700" />
              </div>
              <h3 className="text-slate-400 font-medium mb-2">Ready to Scan</h3>
              <p className="text-sm text-slate-600 max-w-xs">Select an input method on the left and paste your content to begin the hybrid analysis.</p>
            </div>
          )}

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-red-500 mt-0.5" />
              <div>
                <h4 className="text-sm font-bold text-red-400">Analysis Failed</h4>
                <p className="text-xs text-red-300/80 mt-1">{error}</p>
              </div>
            </div>
          )}
        </div>

      </main>
    </div>
  );
}
