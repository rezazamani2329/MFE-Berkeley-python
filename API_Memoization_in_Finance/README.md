# HW5 Submission - ML Prediction API

## Overview
This is a FastAPI application that wraps a trained scikit-learn RandomForestRegressor model and exposes it via a REST API endpoint for predictions.

## ✅ Completion Status
All 5 milestones have been successfully implemented and tested:
- ✅ **Milestone 1**: Model Serialization & API Design
- ✅ **Milestone 2**: Testing (6/6 tests passing)
- ✅ **Milestone 3**: Containerization
- ✅ **Milestone 4**: Automation, Tooling & Readability
- ✅ **Milestone 5**: API Request Testing

## Project Structure
```
.
├── app.py                 # FastAPI application with /predict endpoint
├── model.joblib           # Serialized sklearn model
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker containerization
├── Makefile               # Build and test automation
├── .flake8                # Flake8 linting configuration
├── .isort.cfg             # Import sorting configuration
├── pyproject.toml         # Black and mypy configuration
├── tests/
│   └── test_app.py        # Pytest test suite
└── create_model.py        # Script to generate model.joblib
```

## Installation

### Local Development
```bash
# Create virtual environment (if not already done)
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### Local Development Server
```bash
# Start FastAPI development server
python -m uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Using Makefile
```bash
# Run tests
make test

# Format and lint code
make lint

# Build Docker image
make build

# Run Docker container
make run
```

## API Endpoints

### POST /predict
Accepts input features and returns a model prediction.

**Request:**
```json
{
  "features": [1.0, 2.0, 3.0, 4.0]
}
```

**Response (200 OK):**
```json
{
  "prediction": 0.65
}
```

**Error Responses:**
- **422 Validation Error**: Invalid input format or wrong number of features
- **500 Internal Server Error**: Model inference failed

## Model Details
- **Type**: RandomForestRegressor (scikit-learn)
- **Input Features**: 4 numerical features
- **Output**: Single continuous value (regression)

## Testing

### Run All Tests
```bash
.venv\Scripts\python.exe -m pytest tests/test_app.py -v
```

### Test Results
```
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0 -- C:\Users\ryan\anaconda3\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\ryan
plugins: anyio-4.10.0
collecting ... collected 6 items

test_app.py::test_predict_valid_input PASSED                             [ 16%]
test_app.py::test_predict_invalid_input PASSED                           [ 33%]
test_app.py::test_predict_missing_required_field PASSED                  [ 50%]
test_app.py::test_predict_wrong_data_type PASSED                         [ 66%]
test_app.py::test_predict_model_failure PASSED                           [ 83%]
test_app.py::test_predict_performance PASSED                             [100%]

============================== 6 passed in 4.54s ==============================
```

### Test Coverage
The test suite implements all required tests:

1. **Positive Test** (`test_predict_valid_input`)
   - ✅ Sends valid input with 4 features
   - ✅ Asserts 200 status code
   - ✅ Asserts response contains `prediction` field
   - ✅ Validates response format

2. **Input Validation Tests**
   - ✅ `test_predict_invalid_input`: Wrong number of features → 422
   - ✅ `test_predict_missing_required_field`: Missing required fields → 422
   - ✅ `test_predict_wrong_data_type`: Wrong data type → 422

3. **Model Failure Test** (`test_predict_model_failure`)
   - ✅ Mocks `model.predict` to raise exception
   - ✅ Asserts 500 status code returned
   - ✅ Uses `unittest.mock.patch` for mocking

4. **Performance Test** (`test_predict_performance`)
   - ✅ Measures response latency
   - ✅ Asserts latency < 500ms
   - ✅ Uses `time` module for measurements

### Test Fixtures
- ✅ `valid_input_data()`: Pydantic InputData fixture for reuse
- ✅ `valid_input_dict()`: Dictionary fixture for JSON payloads

## Code Quality

### Linting Configuration
- **Black**: Line length 100, Python 3.12+
- **Flake8**: Maximum line length 100
- **isort**: Import sorting with black profile
- **mypy**: Type checking with ignore_missing_imports

### Running Linters
```bash
make lint
```

## Containerization

### Docker Build and Run
```bash
# Build image (named 'fastapi-app')
make build

# Run container on port 8000
make run

# The API will be available at http://localhost:8000
```

### Dockerfile Details
- Base Image: `python:3.12-slim`
- Working Directory: `/app`
- Exposes: Port 8000
- Command: `uvicorn app:app --host 0.0.0.0 --port 8000`

## Milestone Details

### Milestone 1: Model Serialization & API Design ✅

**Implementation:**
- **Endpoint**: `POST /predict`
- **Input Model**: `InputData` with `features: List[float]`
- **Output Model**: `OutputData` with `prediction: float`
- **Model**: RandomForestRegressor (4 input features) serialized as `model.joblib`

**Features:**
- ✅ Accepts POST requests with JSON body
- ✅ Validates input using Pydantic `InputData` model
- ✅ Loads and uses serialized sklearn model via joblib
- ✅ Returns predictions via Pydantic `OutputData` model
- ✅ Returns 200 OK for valid inputs
- ✅ Returns 422 Validation Error for invalid inputs
- ✅ Returns 500 Internal Server Error for inference failures

### Milestone 2: Testing ✅
See [Test Coverage](#test-coverage) above - all 6 required tests implemented and passing.

### Milestone 3: Containerization ✅

**Dockerfile:**
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
COPY app.py .
COPY model.joblib .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Features:**
- ✅ Sets working directory to `/app`
- ✅ Copies all necessary files
- ✅ Installs dependencies
- ✅ Runs FastAPI with uvicorn on port 8000
- ✅ Configured to accept requests from 0.0.0.0

### Milestone 4: Automation, Tooling & Readability ✅

**Makefile:**
```makefile
test:
	pytest tests/ -v

lint:
	black . --line-length=100
	flake8 . --max-line-length=100
	isort . --profile black
	mypy . --ignore-missing-imports

build:
	docker build -t fastapi-app .

run:
	docker run -d -p 8000:8000 fastapi-app
```

**Linting Configuration:**
- ✅ `.flake8`: Max line length 100, proper exclusions
- ✅ `.isort.cfg`: Black profile, line length 100
- ✅ `pyproject.toml`: Black and mypy configuration

**Code Quality Results:**
- ✅ Black: All files formatted
- ✅ Flake8: No issues found
- ✅ isort: Imports properly sorted
- ✅ mypy: Type checking passed

### Milestone 5: API Request Testing ✅

**Manual Test 1: Valid Request**
```powershell
$body = @{features = @(1.0, 2.0, 3.0, 4.0)} | ConvertTo-Json
Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
```
**Result:** HTTP 200 with prediction value ✅

**Manual Test 2: Invalid Request (wrong feature count)**
```powershell
$body = @{features = @(1.0, 2.0)} | ConvertTo-Json
Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
```
**Result:** HTTP 422 with error message ✅

**Manual Test 3: API Documentation**
```
GET http://127.0.0.1:8000/docs
```
**Result:** Swagger UI loads successfully ✅

---

## Troubleshooting

### Port Already in Use
If port 8000 is already in use:
```bash
# Use a different port with uvicorn
python -m uvicorn app:app --port 8001
```

### Model File Missing
If `model.joblib` is missing:
```bash
python create_model.py
```

## Dependencies
- **fastapi**: Web framework for building APIs
- **uvicorn**: ASGI web server
- **pydantic**: Data validation using Python type annotations
- **scikit-learn**: Machine learning models
- **joblib**: Model serialization
- **pytest**: Testing framework
- **httpx**: HTTP client for testing
- **black**: Code formatter
- **flake8**: Style guide enforcement
- **isort**: Import statement sorting
- **mypy**: Static type checker

## Submission Status

✅ **All Requirements Met:**
- All tests passing (6/6)
- All code properly formatted
- No linting issues
- Type checking passes
- API fully functional and tested
- Docker configuration ready

**Ready for submission!**

## Authors
Submitted for UC Berkeley MFE Python Pre-Program HW5
