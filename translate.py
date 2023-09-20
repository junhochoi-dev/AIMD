# from googletrans import Translator
# translator = Translator()
# print(translator)

# sentence = input("언어를 감지할 문장을 입력해주세요 : ")
# detected = translator.detect(sentence)
# print(detected.lang)

# import googletrans

# translator = googletrans.Translator()

# str1 = "나는 한국인 입니다."
# str2 = "I like burger."
# result1 = translator.translate(str1, dest='en')
# result2 = translator.translate(str2, dest='ko')

# print(f"나는 한국인 입니다. => {result1.text}")
# print(f"I like burger. => {result2.text}")


import googletrans

translator = googletrans.Translator()

def translateK2E(kor):
    return translator.translate(kor, dest='en').text

def translateE2K(eng):
    return translator.translate(eng, dest='ko').text

while True:
    s = input()
    if s == '그만':
        break
    print(translateK2E(s))