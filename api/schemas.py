from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=18, le=100, examples=[54])
    sex: int = Field(ge=0, le=1, description="0=female, 1=male")
    cp: int = Field(ge=1, le=4, description="Chest-pain category (UCI coding 1-4)")
    trestbps: float = Field(ge=70, le=250, description="Resting blood pressure, mm Hg")
    chol: float = Field(ge=80, le=700, description="Serum cholesterol, mg/dl")
    fbs: int = Field(ge=0, le=1, description="Fasting blood sugar >120 mg/dl")
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=50, le=250, description="Maximum heart rate")
    exang: int = Field(ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(ge=-3, le=10, description="Exercise ST depression")
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3, description="Major vessels colored by fluoroscopy")
    thal: int = Field(ge=3, le=7, description="Thal category: 3, 6, or 7")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    model_version: str
    disclaimer: str
