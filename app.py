import os
import re
import cv2
import time
import pickle
import psycopg2
import numpy as np
import psycopg2.extras
import image_database_app
import urllib.parse as urlparse
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
'''
DO NOT UPGRADE opencv-contrib-python and opencv-contrib-python-headless - newer versions do not include SIFT algorithm
'''


# ======================================================================================================================

app = Flask(__name__)

# ======================================================================================================================

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # used to upload images for search


ENV = 'dev'
#ENV = 'prod'

if ENV == 'dev':
    app.debug = True  # Local testing
    # database config
    user = "postgres"
    password = "pass"
    host = "127.0.0.1"
    port = "5432"
    database = "image_db"
else:  # Heroku
    '''
    To upload the project on HEROKU, it is needed to install opencv-contrib-python-headless instead of 
    opencv-contrib-python
    '''
    app.debug = False
    # database config
    url = urlparse.urlparse(os.environ.get('DATABASE_URL'))
    user = url.username
    password = url.password
    host = url.hostname
    port = url.port
    database = url.path[1:]


app.secret_key = "YOUR_SECRET_KEY"


'''
# Define upload folder path to save the uploaded image (create downloaded_images folder in root path of project)
UPLOAD_FOLDER = os.getcwd() + "\\downloaded_images"  # dynamically define downloaded_images folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # define UPLOAD_FOLDER
'''

# ======================================================================================================================

# check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ======================================================================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        descr_or_hist = request.form['descriptor_or_histogram']
        img = ''  # in case no image where given
        img_desc = request.form['image_descriptor']
        distance_metric = request.form['distance_metric']
        k = request.form['get_k_results']

        if request.files:
            imag = request.files['img']
            print(imag)

            # if user does not select file, browser also submit an empty part without filename
            if imag.filename == '' and img_desc == '':
                return render_template('index.html', message='No selected file')
            elif distance_metric == '':
                return render_template('index.html', message='Please select a distance metric')
            elif k == '':
                return render_template('index.html', message='Please select a value for k')

            if imag and allowed_file(imag.filename):
                filename = secure_filename(imag.filename)
                print(filename)

                # read file from buffer instantly (without having to save it and then read it)
                img = cv2.imdecode(np.fromstring(imag.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                '''
                # Save file to 'UPLOAD_FOLDER' and then read it
                img = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print(img)
                imag.save(img)
                img = cv2.imread(img)
                '''
            else:
                if img_desc[0] == '[' and img_desc[-1] == ']':  # if the array is given with [], ignore them
                    img_desc = img_desc[1: -1]
                img_desc = re.sub(r"\s+", "", img_desc)  # delete all white spaces
                # split numbers in the array using comma as seperator and delete dots from floats to check if the input contains only numbers
                if not all(var.replace('.', '', 1).isdigit() for var in img_desc.split(',')):
                    return render_template('index.html', message='Image vector does not contain only numbers')
                print(img_desc)
                img_desc = np.fromstring(img_desc, dtype=float, sep=',')

        print(img, img_desc, distance_metric, k)

        try:
            start_database = time.time()

            conn = psycopg2.connect(user=user,
                                    password=password,
                                    host=host,
                                    port=port,
                                    database=database)

            cur = conn.cursor()
            if descr_or_hist == "sift_descr":  # check if the search is performed with SIFT descriptor or with histogram
                cur.execute("""SELECT image_descriptor, image_link from images""")
                vector_method = "Image vector (SIFT descriptor)"
            elif descr_or_hist == "hist":
                cur.execute("""SELECT image_histogram, image_link from images""")
                vector_method = "Image vector (Histogram)"
            else:
                cur.execute("""SELECT sift_bovw_descriptor, image_link from images""")
                vector_method = "Image vector (SIFT + BOVW)"

            all_db_items = cur.fetchall()

            if descr_or_hist == "sift_bovw":  # get the trained kmeans model from PostgreSQL
                cur.execute("""SELECT kmeans_model from sift_bovw_clustering_model""")
                load_model = cur.fetchone()[0]
                kmeans_model = pickle.loads(bytes(load_model))

            # if user gave an image, calculate its vector with image descriptor
            if len(img):
                if descr_or_hist == "sift_descr":  # check if the search is performed with SIFT descriptor or with histogram
                    img_desc = image_database_app.sift_img_descriptor(img, sift_bovw="sift")
                elif descr_or_hist == "hist":
                    img_desc = image_database_app.histogram(img)
                else:
                    descriptors = image_database_app.sift_img_descriptor(img, sift_bovw="sift_bovw")
                    img_desc = image_database_app.build_histogram(descriptors, kmeans_model)

            print("Data retrieval time: ", time.time() - start_database)
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL: ", error)
            return render_template('index.html', message='"Error while fetching data from PostgreSQL", error')
        finally:
            # closing database connection.
            if conn:
                cur.close()
                conn.close()
                print("PostgreSQL connection is closed")

        start_dist_time = time.time()
        k_similar_images = image_database_app.search_database(img_desc, all_db_items, distance_metric, int((float(k))))
        print("Distance calculation time: ", time.time() - start_dist_time)

        np.set_printoptions(threshold=15)  # while printing an array, truncate its values if there are more than 15

        return render_template('index.html', message='Scroll down for results', results=k_similar_images, vector_method=vector_method)


if __name__ == '__main__':
    app.run()
