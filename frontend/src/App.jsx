import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Gauge,
  RotateCw,
  Thermometer,
  ShieldCheck,
  Cpu,
  RefreshCw,
  Brain,
  FlaskConical,
  TrendingUp,
  TrendingDown,
} from "lucide-react";

import "./App.css";

function App() {
  const defaultMachineData = {
    air_temperature: 298.1,
    process_temperature: 308.6,
    rotational_speed: 1551,
    torque: 42.8,
    tool_wear: 0,
  };

  const [machineData, setMachineData] = useState(
    defaultMachineData
  );

  const [result, setResult] = useState({
    prediction: 0,
    status: "NORMAL",
    failure_probability: 0,
    health_score: 100,
    recommendation:
      "Continue operation and monitor machine parameters.",
  });

  const [explainability, setExplainability] = useState({
    model: "Random Forest Classifier",
    explanation_type: "Global Feature Importance",
    features: [],
  });

  const [whatIfParameter, setWhatIfParameter] =
    useState("torque");

  const [whatIfValue, setWhatIfValue] = useState(42.8);

  const [whatIfResult, setWhatIfResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const [explainabilityLoading, setExplainabilityLoading] =
    useState(true);

  const [whatIfLoading, setWhatIfLoading] = useState(false);

  const [error, setError] = useState("");

  useEffect(() => {
    fetchExplainability();
  }, []);

  const fetchExplainability = async () => {
    setExplainabilityLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/explainability"
      );

      if (!response.ok) {
        throw new Error(
          `Explainability request failed: ${response.status}`
        );
      }

      const data = await response.json();

      setExplainability({
        model:
          data.model || "Random Forest Classifier",
        explanation_type:
          data.explanation_type ||
          "Global Feature Importance",
        features: Array.isArray(data.features)
          ? data.features
          : [],
      });
    } catch (err) {
      console.error("Explainability error:", err);

      setExplainability({
        model: "Random Forest Classifier",
        explanation_type: "Global Feature Importance",
        features: [],
      });
    } finally {
      setExplainabilityLoading(false);
    }
  };

  const handleChange = (event) => {
    const { name, value } = event.target;

    setMachineData((previous) => ({
      ...previous,
      [name]: Number(value),
    }));

    setWhatIfResult(null);
  };

  const analyzeMachine = async () => {
    setLoading(true);
    setError("");
    setWhatIfResult(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(machineData),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Prediction request failed: ${response.status}`
        );
      }

      const data = await response.json();

      setResult({
        prediction: data.prediction ?? 0,
        status: data.status || "NORMAL",
        failure_probability: Number(
          data.failure_probability ?? 0
        ),
        health_score: Number(
          data.health_score ?? 100
        ),
        recommendation:
          data.recommendation ||
          "Continue operation and monitor machine parameters.",
      });

      await fetchExplainability();
    } catch (err) {
      console.error("Prediction error:", err);

      setError(
        "Unable to connect to the AI backend. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetMachine = () => {
    setMachineData(defaultMachineData);

    setWhatIfParameter("torque");

    setWhatIfValue(defaultMachineData.torque);

    setWhatIfResult(null);

    setResult({
      prediction: 0,
      status: "NORMAL",
      failure_probability: 0,
      health_score: 100,
      recommendation:
        "Continue operation and monitor machine parameters.",
    });

    setError("");

    fetchExplainability();
  };

  const getStatusClass = (status = result.status) => {
    if (status === "CRITICAL") {
      return "critical";
    }

    if (status === "WARNING") {
      return "warning";
    }

    return "normal";
  };

  const getFailureRisk = () => {
    const risk = Number(result.failure_probability);

    if (risk < 20) {
      return "Low";
    }

    if (risk < 50) {
      return "Moderate";
    }

    return "High";
  };

  const getFeatureClass = (percentage) => {
    const value = Number(percentage);

    if (value >= 30) {
      return "high";
    }

    if (value >= 15) {
      return "medium";
    }

    return "low";
  };

  const getFeatureInsight = (feature) => {
    if (!feature) {
      return "The model explanation is not currently available.";
    }

    const name = String(feature.name || "").toLowerCase();

    if (name.includes("torque")) {
      return "Torque is the strongest contributing parameter in the trained model, indicating that load-related operating conditions have a major influence on predicted machine failure.";
    }

    if (name.includes("rotational")) {
      return "Rotational speed is a major contributor to the model decision, showing that motor operating speed is strongly associated with machine failure patterns.";
    }

    if (name.includes("tool wear")) {
      return "Tool wear has a significant influence on the prediction, indicating that accumulated operating time and wear conditions should be monitored closely.";
    }

    if (name.includes("air temperature")) {
      return "Air temperature contributes to the model prediction and provides additional information about the machine's thermal operating environment.";
    }

    if (name.includes("process temperature")) {
      return "Process temperature contributes to the model prediction and provides information about the machine's thermal process conditions.";
    }

    return "This parameter contributes to the Random Forest prediction and provides additional information about the current machine operating condition.";
  };

  const getParameterLabel = (parameter) => {
    const labels = {
      air_temperature: "Air Temperature",
      process_temperature: "Process Temperature",
      rotational_speed: "Rotational Speed",
      torque: "Torque",
      tool_wear: "Tool Wear",
    };

    return labels[parameter] || parameter;
  };

  const getParameterUnit = (parameter) => {
    const units = {
      air_temperature: "K",
      process_temperature: "K",
      rotational_speed: "RPM",
      torque: "Nm",
      tool_wear: "min",
    };

    return units[parameter] || "";
  };

  const getParameterValue = (parameter) => {
    return Number(machineData[parameter] ?? 0);
  };

  const handleWhatIfParameterChange = (event) => {
    const parameter = event.target.value;

    setWhatIfParameter(parameter);

    setWhatIfValue(
      getParameterValue(parameter)
    );

    setWhatIfResult(null);
  };

  const handleWhatIfValueChange = (event) => {
    setWhatIfValue(event.target.value);

    setWhatIfResult(null);
  };

  const simulateWhatIf = async () => {
    const numericValue = Number(whatIfValue);

    if (!Number.isFinite(numericValue)) {
      setError("Please enter a valid simulated value.");
      return;
    }

    setWhatIfLoading(true);
    setError("");

    try {
      const simulatedData = {
        ...machineData,
        [whatIfParameter]: numericValue,
      };

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(simulatedData),
        }
      );

      if (!response.ok) {
        throw new Error(
          `What-if prediction failed: ${response.status}`
        );
      }

      const data = await response.json();

      const currentRisk = Number(
        result.failure_probability ?? 0
      );

      const simulatedRisk = Number(
        data.failure_probability ?? 0
      );

      setWhatIfResult({
        ...data,
        currentRisk,
        simulatedRisk,
        riskChange: simulatedRisk - currentRisk,
        parameter: whatIfParameter,
        value: numericValue,
      });
    } catch (err) {
      console.error("What-if error:", err);

      setError(
        "Unable to run the what-if simulation. Make sure FastAPI is running."
      );
    } finally {
      setWhatIfLoading(false);
    }
  };

  const topFeature =
    explainability.features.length > 0
      ? explainability.features[0]
      : null;

  const healthScore = Math.max(
    0,
    Math.min(
      100,
      Number(result.health_score ?? 100)
    )
  );

  const failureProbability = Math.max(
    0,
    Math.min(
      100,
      Number(result.failure_probability ?? 0)
    )
  );

  const currentStatusClass = getStatusClass();

  const whatIfStatusClass = whatIfResult
    ? getStatusClass(whatIfResult.status)
    : "normal";

  return (
    <div className="app">

      <header className="topbar">

        <div className="brand">

          <div className="brand-icon">
            <Cpu size={24} />
          </div>

          <div>
            <h1>EdgeMaintain</h1>

            <p>
              AI-Powered Predictive Maintenance
            </p>
          </div>

        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          System Online
        </div>

      </header>

      <main className="main-content">

        <section className="page-heading">

          <div>
            <h2>Machine Overview</h2>

            <p>
              Monitor machine health and predictive
              maintenance insights.
            </p>
          </div>

          <div className="machine-selector">

            <label>Machine</label>

            <select defaultValue="Industrial Motor 01">
              <option>
                Industrial Motor 01
              </option>

              <option>
                Industrial Motor 02
              </option>

              <option>
                Industrial Motor 03
              </option>
            </select>

          </div>

        </section>

        <section className="summary-grid">

          <div className="summary-card">

            <div className="summary-icon health-icon">
              <Gauge size={27} />
            </div>

            <div>
              <span>Machine Health</span>

              <strong>
                {healthScore.toFixed(0)}%
              </strong>

              <small>
                {healthScore >= 80
                  ? "Excellent condition"
                  : healthScore >= 50
                  ? "Needs attention"
                  : "Critical condition"}
              </small>
            </div>

          </div>

          <div className="summary-card">

            <div className="summary-icon temperature-icon">
              <Thermometer size={27} />
            </div>

            <div>
              <span>Air Temperature</span>

              <strong>
                {machineData.air_temperature} K
              </strong>

              <small>
                Operating temperature
              </small>
            </div>

          </div>

          <div className="summary-card">

            <div className="summary-icon speed-icon">
              <RotateCw size={27} />
            </div>

            <div>
              <span>Rotational Speed</span>

              <strong>
                {machineData.rotational_speed} RPM
              </strong>

              <small>
                Motor speed
              </small>
            </div>

          </div>

          <div className="summary-card">

            <div className="summary-icon torque-icon">
              <Activity size={27} />
            </div>

            <div>
              <span>Torque</span>

              <strong>
                {machineData.torque} Nm
              </strong>

              <small>
                Applied torque
              </small>
            </div>

          </div>

        </section>

        <section className="main-grid">

          <div className="panel">

            <div className="panel-header">

              <div>
                <h3>Machine Health</h3>

                <p>
                  Current operating condition
                </p>
              </div>

              <span
                className={`status-badge ${currentStatusClass}`}
              >
                {result.status}
              </span>

            </div>

            <div className="health-content">

              <div
                className="health-circle"
                style={{
                  "--health": `${healthScore}%`,
                }}
              >

                <div className="health-inner">

                  <strong>
                    {healthScore.toFixed(0)}%
                  </strong>

                  <span>
                    Health
                  </span>

                </div>

              </div>

              <div className="health-details">

                <div className="detail-row">

                  <span>
                    Failure Risk
                  </span>

                  <strong>
                    {getFailureRisk()}
                  </strong>

                </div>

                <div className="detail-row">

                  <span>
                    Prediction Confidence
                  </span>

                  <strong>
                    {(100 - failureProbability).toFixed(1)}%
                  </strong>

                </div>

                <div className="detail-row">

                  <span>
                    Recommended Action
                  </span>

                  <strong>
                    {result.status === "CRITICAL"
                      ? "Immediate Maintenance"
                      : result.status === "WARNING"
                      ? "Inspect Machine"
                      : "Continue Operation"}
                  </strong>

                </div>

              </div>

            </div>

          </div>

          <div className="panel">

            <div className="panel-header">

              <div>
                <h3>AI Prediction</h3>

                <p>
                  Latest model prediction
                </p>
              </div>

              <Activity size={25} />

            </div>

            <div
              className={`prediction-box ${currentStatusClass}`}
            >

              <div className="prediction-icon">

                {result.status === "CRITICAL" ? (
                  <AlertTriangle size={27} />
                ) : result.status === "WARNING" ? (
                  <AlertTriangle size={27} />
                ) : (
                  <ShieldCheck size={27} />
                )}

              </div>

              <div>

                <span>
                  Machine Condition
                </span>

                <strong>
                  {result.status}
                </strong>

                <p>
                  {result.recommendation}
                </p>

              </div>

            </div>

            <div className="probability-section">

              <div className="probability-bar">

                <div
                  className={`probability-fill ${currentStatusClass}`}
                  style={{
                    width: `${failureProbability}%`,
                  }}
                ></div>

              </div>

              <div className="probability-label">

                <span>
                  Failure Probability
                </span>

                <strong>
                  {failureProbability.toFixed(1)}%
                </strong>

              </div>

            </div>

          </div>

        </section>

        <section className="panel whatif-panel">

          <div className="whatif-header">

            <div className="whatif-title">

              <div className="xai-title-icon">
                <FlaskConical size={25} />
              </div>

              <div>

                <h3>
                  What-If Risk Simulator
                </h3>

                <p>
                  Explore how changing a machine
                  parameter affects predicted failure risk.
                </p>

              </div>

            </div>

            <div className="xai-model-badge">
              AI Simulation
            </div>

          </div>

          <div className="whatif-form">

            <div className="whatif-field">

              <label>
                Parameter
              </label>

              <select
                value={whatIfParameter}
                onChange={handleWhatIfParameterChange}
              >

                <option value="air_temperature">
                  Air Temperature
                </option>

                <option value="process_temperature">
                  Process Temperature
                </option>

                <option value="rotational_speed">
                  Rotational Speed
                </option>

                <option value="torque">
                  Torque
                </option>

                <option value="tool_wear">
                  Tool Wear
                </option>

              </select>

            </div>

            <div className="whatif-field">

              <label>
                Simulated Value
              </label>

              <div className="whatif-input-wrapper">

                <input
                  type="number"
                  value={whatIfValue}
                  onChange={handleWhatIfValueChange}
                />

                <span>
                  {getParameterUnit(
                    whatIfParameter
                  )}
                </span>

              </div>

            </div>

            <button
              className="simulate-button"
              onClick={simulateWhatIf}
              disabled={whatIfLoading}
            >

              {whatIfLoading ? (
                <>
                  <RefreshCw
                    size={19}
                    className="spin"
                  />

                  Simulating...
                </>
              ) : (
                <>
                  <FlaskConical size={19} />

                  Simulate Risk
                </>
              )}

            </button>

          </div>

          <div className="whatif-current">

            <div>

              <span>
                Current{" "}
                {getParameterLabel(
                  whatIfParameter
                )}
              </span>

              <strong>
                {getParameterValue(
                  whatIfParameter
                )}{" "}
                {getParameterUnit(
                  whatIfParameter
                )}
              </strong>

            </div>

            <div>

              <span>
                Current Failure Risk
              </span>

              <strong>
                {failureProbability.toFixed(1)}%
              </strong>

            </div>

          </div>

          {whatIfResult && (

            <div className="whatif-result">

              <div className="whatif-result-header">

                <div>

                  <span>
                    Simulated Result
                  </span>

                  <h4>
                    {getParameterLabel(
                      whatIfResult.parameter
                    )}{" "}
                    ={" "}
                    {whatIfResult.value}{" "}
                    {getParameterUnit(
                      whatIfResult.parameter
                    )}
                  </h4>

                </div>

                <span
                  className={`status-badge ${whatIfStatusClass}`}
                >
                  {whatIfResult.status ||
                    "NORMAL"}
                </span>

              </div>

              <div className="risk-comparison">

                <div className="risk-card">

                  <span>
                    Current Risk
                  </span>

                  <strong>
                    {Number(
                      whatIfResult.currentRisk
                    ).toFixed(1)}
                    %
                  </strong>

                </div>

                <div className="risk-arrow">
                  →
                </div>

                <div className="risk-card">

                  <span>
                    Simulated Risk
                  </span>

                  <strong>
                    {Number(
                      whatIfResult.simulatedRisk
                    ).toFixed(1)}
                    %
                  </strong>

                </div>

                <div
                  className={`risk-change ${
                    whatIfResult.riskChange > 0
                      ? "increase"
                      : whatIfResult.riskChange < 0
                      ? "decrease"
                      : "same"
                  }`}
                >

                  {whatIfResult.riskChange > 0 ? (
                    <TrendingUp size={22} />
                  ) : whatIfResult.riskChange < 0 ? (
                    <TrendingDown size={22} />
                  ) : (
                    <Activity size={22} />
                  )}

                  <div>

                    <span>
                      Risk Change
                    </span>

                    <strong>
                      {whatIfResult.riskChange >= 0
                        ? "+"
                        : ""}
                      {Number(
                        whatIfResult.riskChange
                      ).toFixed(1)}
                      %
                    </strong>

                  </div>

                </div>

              </div>

              <div
                className={`whatif-message ${
                  whatIfResult.riskChange > 0
                    ? "danger"
                    : whatIfResult.riskChange < 0
                    ? "safe"
                    : "neutral"
                }`}
              >

                {whatIfResult.riskChange > 0 ? (
                  <AlertTriangle size={21} />
                ) : whatIfResult.riskChange < 0 ? (
                  <ShieldCheck size={21} />
                ) : (
                  <Activity size={21} />
                )}

                <span>

                  {whatIfResult.riskChange > 0
                    ? "Increasing this parameter under the current operating conditions increases the model's predicted failure risk."
                    : whatIfResult.riskChange < 0
                    ? "Changing this parameter to the simulated value decreases the model's predicted failure risk."
                    : "Changing this parameter produces no significant change in the model's predicted failure risk."}

                </span>

              </div>

            </div>

          )}

        </section>

        <section className="panel explainability-panel">

          <div className="panel-header">

            <div className="explainability-heading">

              <div className="xai-title-icon">

                <Brain size={25} />

              </div>

              <div>

                <h3>
                  Explainable AI
                </h3>

                <p>
                  Understand which parameters influence
                  the model prediction.
                </p>

              </div>

            </div>

            <div className="xai-model-badge">

              {explainability.model}

            </div>

          </div>

          {explainabilityLoading ? (

            <div className="xai-loading">

              <RefreshCw
                size={20}
                className="spin"
              />

              Loading model explanation...

            </div>

          ) : explainability.features.length === 0 ? (

            <div className="xai-empty">

              <AlertTriangle size={20} />

              <span>
                Explainability data is currently unavailable.
                Make sure the FastAPI explainability endpoint
                is running.
              </span>

            </div>

          ) : (

            <>

              <div className="xai-content">

                <div className="feature-importance">

                  <div className="xai-section-title">

                    <span>
                      Feature Importance
                    </span>

                    <small>
                      Relative contribution
                    </small>

                  </div>

                  {explainability.features.map(
                    (feature, index) => {

                      const importance = Number(
                        feature.importance ??
                        feature.value ??
                        feature.percentage ??
                        0
                      );

                      const percentage =
                        importance <= 1
                          ? importance * 100
                          : importance;

                      return (
                        <div
                          className="feature-row"
                          key={
                            feature.name ||
                            index
                          }
                        >

                          <div className="feature-label">

                            <div className="feature-rank">
                              {index + 1}
                            </div>

                            <span>
                              {feature.name}
                            </span>

                            <strong>
                              {percentage.toFixed(1)}%
                            </strong>

                          </div>

                          <div className="feature-bar">

                            <div
                              className={`feature-fill ${getFeatureClass(
                                percentage
                              )}`}
                              style={{
                                width: `${Math.min(
                                  100,
                                  percentage
                                )}%`,
                              }}
                            ></div>

                          </div>

                        </div>
                      );
                    }
                  )}

                </div>

                <div className="research-insight">

                  <div className="insight-icon">

                    <Brain size={23} />

                  </div>

                  <div>

                    <span>
                      Top Model Driver
                    </span>

                    <strong>
                      {topFeature
                        ? topFeature.name
                        : "Unavailable"}
                    </strong>

                    <p>
                      {getFeatureInsight(
                        topFeature
                      )}
                    </p>

                  </div>

                </div>

              </div>

              <div className="xai-footer">

                <span>
                  Model explanation:
                </span>

                <strong>
                  {explainability.explanation_type}
                </strong>

              </div>

            </>

          )}

        </section>

        <section className="panel input-panel">

          <div className="panel-header">

            <div>

              <h3>
                Machine Parameters
              </h3>

              <p>
                Enter current operating parameters
                for AI analysis.
              </p>

            </div>

            <button
              className="icon-button"
              onClick={resetMachine}
              title="Reset parameters"
            >

              <RefreshCw size={19} />

            </button>

          </div>

          <div className="input-grid">

            <div className="input-card">

              <div className="input-title">

                <Thermometer size={17} />

                Air Temperature

              </div>

              <input
                type="number"
                name="air_temperature"
                value={machineData.air_temperature}
                onChange={handleChange}
                step="0.1"
              />

              <small>
                Kelvin (K)
              </small>

            </div>

            <div className="input-card">

              <div className="input-title">

                <Thermometer size={17} />

                Process Temperature

              </div>

              <input
                type="number"
                name="process_temperature"
                value={
                  machineData.process_temperature
                }
                onChange={handleChange}
                step="0.1"
              />

              <small>
                Kelvin (K)
              </small>

            </div>

            <div className="input-card">

              <div className="input-title">

                <RotateCw size={17} />

                Rotational Speed

              </div>

              <input
                type="number"
                name="rotational_speed"
                value={
                  machineData.rotational_speed
                }
                onChange={handleChange}
              />

              <small>
                Revolutions per minute (RPM)
              </small>

            </div>

            <div className="input-card">

              <div className="input-title">

                <Activity size={17} />

                Torque

              </div>

              <input
                type="number"
                name="torque"
                value={machineData.torque}
                onChange={handleChange}
                step="0.1"
              />

              <small>
                Newton metres (Nm)
              </small>

            </div>

            <div className="input-card">

              <div className="input-title">

                <Gauge size={17} />

                Tool Wear

              </div>

              <input
                type="number"
                name="tool_wear"
                value={machineData.tool_wear}
                onChange={handleChange}
              />

              <small>
                Tool wear time (minutes)
              </small>

            </div>

          </div>

          <button
            className="analyze-button"
            onClick={analyzeMachine}
            disabled={loading}
          >

            {loading ? (
              <>
                <RefreshCw
                  size={20}
                  className="spin"
                />

                Analyzing Machine...

              </>
            ) : (
              <>
                <Brain size={20} />

                Analyze Machine

              </>
            )}

          </button>

          {error && (

            <div className="error-message">

              <AlertTriangle size={19} />

              <span>
                {error}
              </span>

            </div>

          )}

        </section>

        <section className="panel maintenance-panel">

          <div className="panel-header">

            <div>

              <h3>
                Maintenance Recommendation
              </h3>

              <p>
                AI-generated maintenance guidance
              </p>

            </div>

          </div>

          <div
            className={`maintenance-alert ${currentStatusClass}`}
          >

            <span className="alert-indicator"></span>

            <div>

              <strong>
                {result.status === "CRITICAL"
                  ? "Immediate maintenance required"
                  : result.status === "WARNING"
                  ? "Maintenance inspection recommended"
                  : "Machine operating normally"}
              </strong>

              <p>
                {result.recommendation}
              </p>

            </div>

            <span>
              AI Recommendation
            </span>

          </div>

        </section>

        <div className="model-note">

          <span>
            EdgeMaintain AI • Random Forest Predictive
            Maintenance Model
          </span>

        </div>

      </main>

    </div>
  );
}

export default App;