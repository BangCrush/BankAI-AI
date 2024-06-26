import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from flask import Flask, request, jsonify, make_response
from flask_restx import Resource, Api
from flask_cors import CORS
from werkzeug.utils import secure_filename
from SpeachToAction import STT, TTA,test
from Hugging_Face_kobert.similarity_calculator import find_most_similar_word
import json



app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)
api = Api(app)




# @api.route('/sentence/insert')
# class Sentence(Resource):
#     def post(self):
#         gender = request.json.get('gender')
#         input_text = request.json.get('input_text')
#         input_image = '../Wav2Lip/my_data/'+request.json.get('character')
#         out_path = request.json.get('out_path')
#         filename = request.json.get('filename')
#         return jsonify({'success': generate_lipsync.generate(gender,input_text,input_image,out_path,filename),'path':out_path+filename});

# @api.route('/requsest')
# class OCR(Resource):
#     def post(self):
#         file = request.json.get('filename')
#         return ocr_api.ocr_api(file);
    
# @api.route('/request')
# class STA(Resource):
#     def post(self):
#         if 'voice' not in request.files:
#             return jsonify({'error': 'No voice part'})
#         file = request.files['voice']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'})
#         if file.filename != secure_filename(file.filename):
#             return jsonify({'error': 'Invalid file name'})
#         return jsonify(STT.stt(file))


def readAzure():
    with open(os.getcwd() + '/Config/Azure.txt', 'r', encoding='utf-8') as f:
        txt_data = f.read()
        data = txt_data.split('\n')

    return data[0], data[1]

@api.route('/test')
class TEST(Resource):
    def get(self):
        return jsonify({'test': 'test1'})

@api.route('/request')
class Request(Resource):
    def post(self):
        # Azure key, region
        azure_key, azure_region = readAzure()

        # >>>>> 음성 파일 사용 <<<<<
        # if 'voice' not in request.files:
        #     return jsonify({'error': 'No voice part'})
        # file = request.files['voice']
        # if file.filename == '':
        #     return jsonify({'error': 'No selected file'})
        # if file.filename != secure_filename(file.filename):
        #     return jsonify({'error': 'Invalid file name'})
        
        # # 음성 파일 저장
        # audio_file_path = os.path.join(os.getcwd(), "audio_files", file.filename)
        # file.save(audio_file_path)

        # # 음성인식 값
        # result = jsonify(STT.sttAzure(azure_key, azure_region, audio_file_path))

        # # 음성 파일 삭제
        # os.remove(audio_file_path)


        # >>>>> 마이크 사용 <<<<<
        data = request.get_json()
        
        text = data.get('text')
        options = data.get('options')
        print(options)
        type = data.get('type')
        stt_result = STT.from_mic(text,type)
        if type == "number":
            result = {'result': stt_result}
            result = json.dumps(result, ensure_ascii=False)
            res = make_response(result)
            return res
        references = []
        for option in options:
            references.append(option['name'])
        result = find_most_similar_word(stt_result,references)
        result = json.dumps(result, ensure_ascii=False)
        print(result)
        res = make_response(result)
        return res

@api.route('/stt/test')
class STTTest(Resource):
    def post(self):
        if 'voice' not in request.files:
            return jsonify({'error': 'No voice part'})
        file = request.files['voice']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file.filename != secure_filename(file.filename):
            return jsonify({'error': 'Invalid file name'})
        return jsonify(STT.stt(file))

@api.route('/stt/test')
class STTTest(Resource):
    def post(self):
        if 'voice' not in request.files:
            return jsonify({'error': 'No voice part'})
        file = request.files['voice']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file.filename != secure_filename(file.filename):
            return jsonify({'error': 'Invalid file name'})
        return jsonify(STT.stt(file))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 35281)))