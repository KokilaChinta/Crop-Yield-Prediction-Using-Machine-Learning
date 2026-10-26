
from pathlib import Path
import joblib

model_path = Path(__file__).resolve().parent / "random_forest_crop_yield.pkl"
model = joblib.load(model_path)

print("Model type:", type(model))

if hasattr(model, "named_steps"):
    print("Pipeline steps:", list(model.named_steps.keys()))

    for name, step in model.named_steps.items():
        print("\nStep:", name)
        print("Type:", type(step))

        if hasattr(step, "feature_names_in_"):
            print("Input features:", list(step.feature_names_in_))

        if hasattr(step, "get_feature_names_out"):
            try:
                print("Output features:", list(step.get_feature_names_out()))
            except Exception as error:
                print("Could not get output features:", error)
