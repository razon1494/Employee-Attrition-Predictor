import gradio as gr
import pandas as pd
import numpy as np
import pickle

# IMPORTANT: import custom transformers BEFORE loading pickle
from transformers import RatioFeatureEngineer, IQRClipper  # keep your exact names

BUNDLE_PATH = "employee_attrition_bundle.pkl"

with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
FEATURE_COLS = bundle["feature_cols"]
CAT_COLS = bundle.get("cat_cols", [])
NUM_COLS = bundle.get("num_cols", [])
CATEGORY_CHOICES = bundle.get("category_choices", {})

LABEL_MAP = {
    "Age": "Age",
    "BusinessTravel": "Business Travel",
    "DailyRate": "Daily Rate",
    "Department": "Department",
    "DistanceFromHome": "Distance From Home (miles)",
    "Education": "Education Level",
    "EducationField": "Education Field",
    "EnvironmentSatisfaction": "Environment Satisfaction",
    "Gender": "Gender",
    "HourlyRate": "Hourly Rate",
    "JobInvolvement": "Job Involvement",
    "JobLevel": "Job Level",
    "JobRole": "Job Role",
    "JobSatisfaction": "Job Satisfaction",
    "MaritalStatus": "Marital Status",
    "MonthlyIncome": "Monthly Income ($)",
    "MonthlyRate": "Monthly Rate",
    "NumCompaniesWorked": "Num Companies Worked",
    "OverTime": "OverTime",
    "PercentSalaryHike": "Percent Salary Hike (%)",
    "PerformanceRating": "Performance Rating",
    "RelationshipSatisfaction": "Relationship Satisfaction",
    "StockOptionLevel": "Stock Option Level",
    "TotalWorkingYears": "Total Working Years",
    "TrainingTimesLastYear": "Training Times Last Year",
    "WorkLifeBalance": "Work-Life Balance",
    "YearsAtCompany": "Years At Company",
    "YearsInCurrentRole": "Years In Current Role",
    "YearsSinceLastPromotion": "Years Since Last Promotion",
    "YearsWithCurrManager": "Years With Current Manager",
}

CODED_CHOICES = {
    "Education": [
        ("1 — Below College", 1),
        ("2 — College", 2),
        ("3 — Bachelor", 3),
        ("4 — Master", 4),
        ("5 — Doctorate", 5),
    ],
    "EnvironmentSatisfaction": [
        ("1 — Low", 1), ("2 — Medium", 2), ("3 — High", 3), ("4 — Very High", 4)
    ],
    "JobInvolvement": [
        ("1 — Low", 1), ("2 — Medium", 2), ("3 — High", 3), ("4 — Very High", 4)
    ],
    "JobSatisfaction": [
        ("1 — Low", 1), ("2 — Medium", 2), ("3 — High", 3), ("4 — Very High", 4)
    ],
    "RelationshipSatisfaction": [
        ("1 — Low", 1), ("2 — Medium", 2), ("3 — High", 3), ("4 — Very High", 4)
    ],
    "WorkLifeBalance": [
        ("1 — Bad", 1), ("2 — Good", 2), ("3 — Better", 3), ("4 — Best", 4)
    ],
    "PerformanceRating": [
        ("1 — Low", 1), ("2 — Good", 2), ("3 — Excellent", 3), ("4 — Outstanding", 4)
    ],
    "JobLevel": [
        ("1 — Entry", 1), ("2 — Junior", 2), ("3 — Mid", 3), ("4 — Senior", 4), ("5 — Executive", 5)
    ],
}

SECTIONS = ["Personal", "Job", "Compensation", "Satisfaction", "Experience", "Review"]

SECTION_MAP = {
    "Personal": {"Age", "Gender", "MaritalStatus", "Education", "EducationField", "DistanceFromHome"},
    "Job": {"Department", "BusinessTravel", "JobRole", "OverTime", "JobLevel"},
    "Compensation": {
        "MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate",
        "PercentSalaryHike", "StockOptionLevel"
    },
    "Satisfaction": {
        "EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
        "WorkLifeBalance", "JobInvolvement"
    },
    "Experience": {
        "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager", "NumCompaniesWorked",
        "TrainingTimesLastYear", "PerformanceRating"
    },
}

def get_section_for_feature(col: str) -> str:
    for sec, cols in SECTION_MAP.items():
        if col in cols:
            return sec
    return "Job"

def risk_band(prob_yes: float) -> str:
    if prob_yes >= 0.70:
        return "🔴 High Risk"
    elif prob_yes >= 0.40:
        return "🟠 Medium Risk"
    return "🟢 Low Risk"

def render_progress(step_idx: int) -> str:
    total = len(SECTIONS)
    return f"### Step {step_idx+1} of {total}: **{SECTIONS[step_idx]}**"

def make_input_df(values):
    row = {c: v for c, v in zip(FEATURE_COLS, values)}
    df = pd.DataFrame([row], columns=FEATURE_COLS)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_review_table(values):
    rows = []
    for c, v in zip(FEATURE_COLS, values):
        rows.append([LABEL_MAP.get(c, c), v])
    return pd.DataFrame(rows, columns=["Feature", "Value"])

def predict_attrition(*vals):
    input_df = make_input_df(vals)
    pred = int(model.predict(input_df)[0])
    label = "Yes" if pred == 1 else "No"

    if hasattr(model, "predict_proba"):
        prob_yes = float(model.predict_proba(input_df)[0][1])
        return (
            f"**Attrition Prediction:** {label}\n\n"
            f"**Probability (Attrition=Yes):** {prob_yes*100:.2f}%\n\n"
            f"**Risk Band:** {risk_band(prob_yes)}"
        )

    return f"**Attrition Prediction:** {label}"

def set_step(step_idx: int):
    step_idx = int(step_idx)
    step_idx = max(0, min(step_idx, len(SECTIONS)-1))
    vis_updates = [gr.update(visible=(i == step_idx)) for i in range(len(SECTIONS))]
    return [render_progress(step_idx), step_idx] + vis_updates

def next_step(step_idx: int):
    return set_step(min(int(step_idx) + 1, len(SECTIONS) - 1))

def prev_step(step_idx: int):
    return set_step(max(int(step_idx) - 1, 0))

# ---------- CSS for colorful buttons ----------
CUSTOM_CSS = """
#nav_back button {
  background: #2563eb !important;   /* blue */
  color: white !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
}
#nav_next button {
  background: #f97316 !important;   /* orange */
  color: white !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
}
#nav_back button:hover { filter: brightness(0.95); }
#nav_next button:hover { filter: brightness(0.95); }
"""

with gr.Blocks(title="Employee Attrition Risk Predictor", css=CUSTOM_CSS) as demo:
    gr.Markdown("# Employee Attrition Risk Predictor")
    gr.Markdown("Complete the form step-by-step using **Next / Back**. Final step: **Predict**.")

    step_state = gr.State(0)
    progress_md = gr.Markdown(render_progress(0))

    input_components = {}
    section_boxes = []

    # placeholders for Review step widgets
    review_table = None
    result_md = None
    predict_btn = None
    clear_btn = None

    for sec in SECTIONS:
        with gr.Column(visible=(sec == "Personal")) as box:
            section_boxes.append(box)
            gr.Markdown(f"## {sec}")

            if sec != "Review":
                cols = [c for c in FEATURE_COLS if get_section_for_feature(c) == sec]
                with gr.Row():
                    with gr.Column():
                        for c in cols[0::2]:
                            label = LABEL_MAP.get(c, c)
                            if c in CAT_COLS:
                                choices = CATEGORY_CHOICES.get(c, [])
                                default = choices[0] if choices else None
                                input_components[c] = gr.Dropdown(label=label, choices=choices, value=default)
                            elif c in CODED_CHOICES:
                                input_components[c] = gr.Dropdown(label=label, choices=CODED_CHOICES[c], value=CODED_CHOICES[c][0][1])
                            else:
                                input_components[c] = gr.Number(label=label, value=0, precision=0)

                    with gr.Column():
                        for c in cols[1::2]:
                            label = LABEL_MAP.get(c, c)
                            if c in CAT_COLS:
                                choices = CATEGORY_CHOICES.get(c, [])
                                default = choices[0] if choices else None
                                input_components[c] = gr.Dropdown(label=label, choices=choices, value=default)
                            elif c in CODED_CHOICES:
                                input_components[c] = gr.Dropdown(label=label, choices=CODED_CHOICES[c], value=CODED_CHOICES[c][0][1])
                            else:
                                input_components[c] = gr.Number(label=label, value=0, precision=0)

            else:
                gr.Markdown("## Review your inputs (all fields)")
                review_table = gr.Dataframe(
                    headers=["Feature", "Value"],
                    value=pd.DataFrame(columns=["Feature", "Value"]),
                    interactive=False,
                    wrap=True
                )

                gr.Markdown("## Predict")
                with gr.Row():
                    predict_btn = gr.Button("🔍 Predict", variant="primary")
                    clear_btn = gr.Button("🧹 Clear All", variant="secondary")
                result_md = gr.Markdown(value="")

    # Navigation buttons with IDs for CSS
    with gr.Row():
        back_btn = gr.Button("⬅ Back", elem_id="nav_back")
        next_btn = gr.Button("Next ➡", elem_id="nav_next")

    back_btn.click(fn=prev_step, inputs=step_state, outputs=[progress_md, step_state] + section_boxes)
    next_btn.click(fn=next_step, inputs=step_state, outputs=[progress_md, step_state] + section_boxes)

    # When entering Review page, refresh the table
    def refresh_review(step_idx, *vals):
        step_idx = int(step_idx)
        if SECTIONS[step_idx] == "Review":
            return make_review_table(vals)
        return gr.update()

    # This triggers whenever step changes (Back/Next)
    step_state.change(
        fn=refresh_review,
        inputs=[step_state] + [input_components[c] for c in FEATURE_COLS],
        outputs=review_table
    )

    # Predict: also refresh table + show prediction
    def predict_and_show(*vals):
        table = make_review_table(vals)
        result = predict_attrition(*vals)
        return table, result

    predict_btn.click(
        fn=predict_and_show,
        inputs=[input_components[c] for c in FEATURE_COLS],
        outputs=[review_table, result_md]
    )

    # Clear: reset inputs + go back to step 0
    def clear_all():
        updates = []
        for c in FEATURE_COLS:
            if c in CAT_COLS:
                choices = CATEGORY_CHOICES.get(c, [])
                default = choices[0] if choices else None
                updates.append(gr.update(value=default))
            elif c in CODED_CHOICES:
                updates.append(gr.update(value=CODED_CHOICES[c][0][1]))
            else:
                updates.append(gr.update(value=0))

        # reset step view
        base = set_step(0)
        empty_table = pd.DataFrame(columns=["Feature", "Value"])
        return [empty_table, ""] + base + updates

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[review_table, result_md, progress_md, step_state] + section_boxes + [input_components[c] for c in FEATURE_COLS]
    )

    demo.load(fn=set_step, inputs=step_state, outputs=[progress_md, step_state] + section_boxes)

if __name__ == "__main__":
    demo.launch()
