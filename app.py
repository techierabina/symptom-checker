from transformers import pipeline
import gradio as gr

print("Loading model... please wait")
classifier = pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-MiniLM2-L6-H768"
)
print("Model loaded!")

RED_FLAGS = [
    "chest pain", "cant breathe", "can't breathe", "difficulty breathing",
    "numbness", "cant move", "unconscious", "seizure",
    "stroke", "heart attack", "bleeding", "crushing",
    "radiating", "911", "emergency", "severe pain"
]

VAGUE_INPUTS = [
    "i feel bad", "not good", "feeling off",
    "feel weird", "something wrong", "i feel sick"
]

MODERATE_FLAGS = [
    "fever", "vomiting", "can't eat", "can't sleep",
    "body aches", "chills", "sore throat", "ear pain",
    "mild nausea", "dizziness", "fatigue", "congestion"
]

def check_symptoms(symptoms):
    if not symptoms or symptoms.strip() == "":
        return "⚠️ Please describe your symptoms first."

    text = symptoms.lower().strip()

    if any(vague in text for vague in VAGUE_INPUTS) or len(text.split()) < 4:
        return """
        
**⚠️ Could you be more specific?**
Try describing:
- What part of your body is affected?
- How long have you had these symptoms?
- How intense is the discomfort (1–10)?

*Example: "I have a headache behind my eyes for 2 days, pain level 4"*
"""

    has_red_flag = any(flag in text for flag in RED_FLAGS)
    if has_red_flag:
        return """
**Severity: 🚨 SEVERE**
**Reason: Red flag symptom detected**

**Recommended action:** Seek emergency care immediately. Call 911 if needed.

---
⚠️ *This is a student demo, not real medical advice. Always consult a doctor.*
"""

    has_moderate_flag = any(flag in text for flag in MODERATE_FLAGS)
    if has_moderate_flag:
        return """
**Severity: ⚠️ MODERATE**
**Reason: Moderate symptom detected**

**Recommended action:** Consider seeing a doctor within 24-48 hours.

---
⚠️ *This is a student demo, not real medical advice. Always consult a doctor.*
"""

    # Run the model for everything else
    labels = ["mild symptoms", "moderate symptoms", "severe symptoms"]
    result = classifier(symptoms, candidate_labels=labels)
    top_label = result["labels"][0]
    top_score = round(result["scores"][0] * 100, 1)

    # If model isn't confident enough, default to MODERATE
    if top_score < 70:
        return f"""
**Severity: ⚠️ MODERATE**
**Confidence: {top_score}% (low — please describe symptoms in more detail)**

**Recommended action:** Consider seeing a doctor if symptoms persist.

---
⚠️ *This is a student demo, not real medical advice. Always consult a doctor.*
"""

    advice = {
        "mild symptoms":    ("✅ MILD",     "Rest at home, stay hydrated, monitor your symptoms."),
        "moderate symptoms":("⚠️ MODERATE", "Consider seeing a doctor within 24-48 hours."),
        "severe symptoms":  ("🚨 SEVERE",   "Seek emergency care immediately. Call 911 if needed.")
    }

    level, action = advice[top_label]

    return f"""
**Severity: {level}**
**Confidence: {top_score}%**

**Recommended action:** {action}

---
⚠️ *This is a student demo, not real medical advice. Always consult a doctor.*
"""

with gr.Blocks(theme=gr.themes.Soft(font=gr.themes.GoogleFont("Poppins"))) as app:
    gr.Markdown("# 🩺 Symptom Severity Checker")
    gr.Markdown("Type your symptoms in plain English. AI assesses severity and suggests next steps.")

    with gr.Row():
        with gr.Column():
            symptoms_input = gr.Textbox(
                lines=4,
                placeholder="Describe your symptoms... e.g. 'I have a fever and body aches'",
                label="Your Symptoms"
            )

            gr.Markdown("**Try an example:**")
            btn1 = gr.Button("✅ Slight headache and tired")
            btn2 = gr.Button("⚠️ Fever 101 and body aches")
            btn3 = gr.Button("🚨 Chest pain, difficulty breathing")
            btn4 = gr.Button("❓ I feel bad")
            btn5 = gr.Button("🚨 Mild headache but arm numbness")

            submit_btn = gr.Button("Check Symptoms", variant="primary")

        with gr.Column():
            output = gr.Markdown(label="Assessment")

    btn1.click(fn=lambda: "I have a slight headache and feel a little tired", outputs=symptoms_input)
    btn2.click(fn=lambda: "Fever of 101, body aches, and can't get out of bed", outputs=symptoms_input)
    btn3.click(fn=lambda: "Severe chest pain radiating to my left arm, difficulty breathing", outputs=symptoms_input)
    btn4.click(fn=lambda: "I feel bad", outputs=symptoms_input)
    btn5.click(fn=lambda: "Mild headache but also sudden numbness in my left arm", outputs=symptoms_input)

    submit_btn.click(fn=check_symptoms, inputs=symptoms_input, outputs=output)

if __name__ == "__main__":
    app.launch()