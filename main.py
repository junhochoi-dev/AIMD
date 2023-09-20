import gradio as gr
import torch
import transformers
import googletrans
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

translator = googletrans.Translator()

def translateK2E(kor):
    return translator.translate(kor, dest='en').text

def translateE2K(eng):
    return translator.translate(eng, dest='ko').text

model_name = "./ChatDoctor_weights/"

tokenizer = transformers.LLaMATokenizer.from_pretrained(model_name)
model = transformers.LLaMAForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_8bit=False, 
    cache_dir="cache"
).cuda()

generator = model.generate

feature_extractor_brain = AutoFeatureExtractor.from_pretrained("Devarshi/Brain_Tumor_Classification")
model_brain = AutoModelForImageClassification.from_pretrained("Devarshi/Brain_Tumor_Classification")

feature_extractor_lung = AutoFeatureExtractor.from_pretrained("hamdan07/UltraSound-Lung")
model_lung = AutoModelForImageClassification.from_pretrained("hamdan07/UltraSound-Lung")

def bmi(name, height, weight, feeling):
    bmi_val = weight / ((height * 0.01) ** 2)
    if bmi_val <= 18.5:
        result_emoticon = "You are underweight"
        if feeling:
            txt = "You should eat properly"
        else:
            txt = "Don't worry too much, just eat properly"
    elif bmi_val <= 24.9:
        result_emoticon = "You are healthy"
        if feeling:
            txt = "Fine no problem"
        else:
            txt = "Your BMI is ok, you may have another problem"
    else:
        result_emoticon = "You are overweight"
        if feeling:
            txt = "You should be worried, You need to take care of your weight."
        else:
            txt = "You need to take care of your weight."

    output_str = "Hello " + name + "\nyour BMI is " + str(bmi_val) + '!!'

    return (output_str, result_emoticon, txt)

def classify_image_brain(image):
    inputs = feature_extractor_brain(images=image, return_tensors="pt")
    outputs = model_brain(**inputs)
    probabilities = outputs.logits.softmax(dim=-1)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    confidences = {model_brain.config.id2label[i]: float(probabilities.tolist()[0][i]) for i in range(4)}
    return confidences

def classify_image_lung(image):
    inputs = feature_extractor_lung(images=image, return_tensors="pt")
    outputs = model_lung(**inputs)
    probabilities = outputs.logits.softmax(dim=-1)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    confidences = {model_lung.config.id2label[i]: float(probabilities.tolist()[0][i]) for i in range(3)}
    return confidences

def answer(state, state_chatbot, text):
    state = state + [f"Patient: {text}"]

    fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\n" + "\n\n".join(state) + "\n\n" + "ChatDoctor: "

    generated_text = ""
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        generated_ids = generator(
            gen_in,
            max_new_tokens=200,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
            temperature=0.5, # default: 1.0
            top_k = 50, # default: 50
            top_p = 1.0, # default: 1.0
            early_stopping=True,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

        text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt
    response = response.split("Patient: ")[0].strip()

    state = state + [f"ChatDoctor: {response}"]
    state_chatbot = state_chatbot + [(text, response)]

    return state, state_chatbot, state_chatbot

with gr.Blocks(css='#chatbot .overflow-y-auto{height:1500px}') as demo:
    state = gr.State(["ChatDoctor: I am ChatDoctor, what medical questions do you have?"])
    state_chatbot = gr.State([(None, "I am ChatDoctor, what medical questions do you have?")])

    with gr.Column():
        gr.HTML("""<div style="text-align: center; margin: 0 auto;">
            <div>
                <h1>Capstone Design Project : AI Medical Diagnosis</h1>
            </div>
            <p style="margin-bottom: 10px; font-size: 94%">
                <a href="https://github.com/junhochoi-dev">Github Link</a>
            </p>
        </div>""")
        gr.Markdown("""
        ##### 컴퓨터융합소프트웨어학과 2018270667
        ##### 최  준  호
        """)
        gr.HTML("""<hr/>""")
        with gr.Row():
            gr.Image("https://t3.ftcdn.net/jpg/04/52/82/90/360_F_452829077_JU5RhSmKQHYSi4mPpMaPIZweDiuIessf.jpg")
            gr.Markdown("""
            * Caution *
            ## The diagnosis of this project is not 100% reliable.
            ## Get professional consultation from a specialist.
            ## In case of an emergency, go to the emergency room no matter what.
            """)
        gr.HTML("""<hr/>""")

    with gr.Row().style(equal_height=True):
        gr.Markdown("""
        # BMI Diagnosis
        Enter your weight, height value.
        """)

    with gr.Column():
        with gr.Row():
            with gr.Column():
                inputs_bmi = [gr.inputs.Textbox(label="name"),
                            gr.inputs.Slider(100.00, 200.0, label="Hieght in CM"),
                            gr.inputs.Slider(30, 200, label="Weight in KG"),
                            gr.inputs.Checkbox(default=0, label="Are you feeling good today ?")]

            with gr.Column():
                outputs_bmi = [gr.outputs.Textbox(label="BMI Value"),
                            gr.outputs.Textbox(label="BMI Result"),
                            gr.outputs.Textbox(label="Advice")]
        btn_bmi = gr.Button("Submit")
        btn_bmi.click(bmi, inputs=inputs_bmi, outputs=outputs_bmi)

    with gr.Row().style(equal_height=True):
        gr.Markdown("""
        # Brain Diagnosis
        Enter a scan image of the Brain for diagnosis.
        """)

    with gr.Column():
        with gr.Row():  
            inputs_brain = gr.Image() 
            outputs_brain = gr.Label(num_top_classes=4)
        btn_brain = gr.Button("Submit")
        btn_brain.click(classify_image_brain, inputs=inputs_brain, outputs=outputs_brain)

    with gr.Row().style(equal_height=True):
        gr.Markdown("""
        # Lung Diagnosis
        Enter a scan image of the Lung for diagnosis.
        """)

    with gr.Column():
        with gr.Row():
            inputs_lung = gr.Image()
            outputs_lung = gr.Label(num_top_classes=3)
        btn_lung = gr.Button("Submit")
        btn_lung.click(classify_image_lung, inputs=inputs_lung, outputs=outputs_lung)

    with gr.Row().style(equal_height=True):
        gr.Markdown("""
        # AI Doctor
        Get an AI Doctor to diagnose anything about your health.
        """)

    with gr.Row().style(equal_height=True):
        with gr.Column():
            gr.Image("https://image.ytn.co.kr/general/jpg/2023/0227/202302271539446745_t.jpg")

        with gr.Column():
            chatbot = gr.Chatbot([(None, "I am ChatDoctor, what medical questions do you have?")], elem_id='chatbot')
            txt = gr.Textbox(show_label=False, placeholder='Please explain your uncomfortable body parts in English').style(container=False)

    txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot])
    txt.submit(lambda: '' , None, txt)
    
demo.launch(debug=True, share=True)