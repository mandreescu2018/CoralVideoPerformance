# import the necessary packages
from flask import Flask, render_template, Response, redirect
from flask import request, jsonify
from test_video import VideoClipCoral
from config import config
from config import settings
import os

app = Flask(__name__)

video_name = 'SJTU-SEIEE-170_Walk_0006'
app.selected_video = video_name

def get_video():
    return app.selected_video

def get_video_files():
    files = os.listdir(config.VIDEO_PATH.format(''))
    dictFiles = {i: files[i].rsplit('.', 1)[0] for i in range(0, len(files))}
    print(dictFiles[2])
    return dictFiles

video_files = get_video_files()
# video_files = ['SJTU-SEIEE-170_Walk_0006']

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    func()
    return "Quitting..."


@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html', video_files=video_files, selectedvideo=app.selected_video)


def gen(video):
    while True:
        # get camera frame
        frame = video.get_frame()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoClipCoral(app.selected_video)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result', methods=['POST'])
def results():
    str_res = str(settings.inference_result)
    return jsonify({'output':'Result: ' + str_res})


@app.route('/playvideo', methods=['GET', 'POST'])
def connect_management():
    video_id = request.form.get('selected_class')
    print('video:', video_files[int(video_id)])
    app.selected_video = video_files[int(video_id)]
    return redirect('/')


if __name__ == '__main__':
    settings.init()
    # defining server ip address and port
    app.run(host='0.0.0.0', port=5000, debug=True)