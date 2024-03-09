from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from photo_restorer import predict_img 
from photo_restorer import deoldify
# from photo_restorer import deoldify1
from photo_restorer import is_black_and_white

# from photo_restorer import deoldify_image # Adjust this import based on your module's content

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

@app.route("/")
def home():
    return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            full_filename = UPLOAD_FOLDER + '/' + filename
            file.save(full_filename)
            # is_bw = is_black_and_white(full_filename)

            # if is_bw:
            restored_img_url =  predict_img(full_filename)
            # print(restored_img_url)
            restored_img_url1 = deoldify(restored_img_url)
            # else:
            #     restored_img_url = predict_img(full_filename)
            #     restored_img_url1 ='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRgTJyxb67k7Hr2uwVIOBZlPL6FqNdN3n-5iA&usqp=CAU'
            #     print("Is Black and White:", is_bw)
                

            return render_template("index.html", filename=filename, restored_img_url=restored_img_url,restored_img_url1=restored_img_url1)
@app.route('/upload_mask', methods=['POST'])
def upload_mask():
    if request.method == 'POST':
        if 'mask' not in request.files:
            flash('No mask file found')
            return redirect(request.url)
        mask_file = request.files['mask']
        if mask_file.filename == '':
            flash('No mask file selected')
            return redirect(request.url)

        if mask_file and allowed_file(mask_file.filename):
            mask_filename = secure_filename(mask_file.filename)
            full_mask_path = UPLOAD_FOLDER + '/' + mask_filename
            mask_file.save(full_mask_path)

            # Perform inpainting and restoration using the uploaded mask
            
                
            restored_img_url, inpainted_img_url = inpaint_and_restore_with_mask(full_filename, full_mask_path)
            return render_template("index.html", filename=filename, restored_img_url=restored_img_url, inpainted_img_url=inpainted_img_url)
        
if __name__ == "__main__":
    app.run(debug=True)
