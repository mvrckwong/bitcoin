-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    actual_price DECIMAL(20, 2),
    predicted_price DECIMAL(20, 2),
    absolute_error DECIMAL(20, 2),
    percentage_error DECIMAL(10, 4),
    direction_actual VARCHAR(10),
    direction_predicted VARCHAR(10),
    direction_correct BOOLEAN,
    within_threshold BOOLEAN,
    success_status VARCHAR(10),
    confidence_score DECIMAL(5, 4),
    model_name VARCHAR(100),
    prediction_horizon VARCHAR(10),
    features_used TEXT,
    sequence_length INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_success_status ON predictions(success_status);

-- Add any additional indexes or constraints as needed 