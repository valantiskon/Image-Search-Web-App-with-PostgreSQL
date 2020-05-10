import os
import cv2
import json
import pickle
import psycopg2
import psycopg2.extras
import image_database_app
import urllib.parse as urlparse
'''
DO NOT UPGRADE opencv-contrib-python and opencv-contrib-python-headless - newer versions do not include SIFT algorithm
'''

ENV = 'dev'
#ENV = 'prod'

if ENV == 'dev':
    # database config
    user = "postgres"
    password = "pass"
    host = "127.0.0.1"
    port = "5432"
    database = "image_db"
else:  # Heroku
    '''
        To upload the data on the PostgreSQL used by HEROKU, it is needed to use hard coded database credentials
    '''
    # database config
    user = "user"
    password = "password"
    host = "host"
    port = "port"
    database = "database"
    '''
    url = urlparse.urlparse(os.environ.get('DATABASE_URL'))
    user = url.username
    password = url.password
    host = url.hostname
    port = url.port
    database = url.path[1:]
    '''



def vectorize_images():
    global connection
    path = os.getcwd() + "\\static\\images"

# ======================================================================================================================
    # Calculate SIFT with BOVW
    # iterate through the names of contents of the folder
    descriptor_vocabulary = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("static\\images", image_name)
        image = cv2.imread(input_path)

        descriptors = image_database_app.sift_img_descriptor(image, sift_bovw="sift_bovw")
        descriptor_vocabulary.extend(descriptors)

    print("KMEANS TRAINING")
    print("Vocabulary size: ", len(descriptor_vocabulary))

    # Takes the central points which is visual words
    kmeans_model = image_database_app.kmeans_sift_bovw(descriptor_vocabulary)
# ======================================================================================================================

    try:
        connection = psycopg2.connect(user=user,
                                      password=password,
                                      host=host,
                                      port=port,
                                      database=database)
        cursor = connection.cursor()

        # (ARRAY[%s],ARRAY[%s])
        postgres_insert_query = """INSERT INTO images (image_descriptor, image_link, image_histogram, sift_bovw_descriptor) VALUES (%s,%s,%s,%s)"""
    except (Exception, psycopg2.Error) as error:
        if connection:
            print("Failed to insert record into images table:", error)


    # iterate through the names of contents of the folder
    for image_name in os.listdir(path):
        print("=======================================================================================================")
        # create the full input path and read the file
        input_path = os.path.join("static\\images", image_name)
        image = cv2.imread(input_path)
        #print(image)
        print(input_path)

        img_vec = image_database_app.sift_img_descriptor(image, sift_bovw="sift")
        img_hist = image_database_app.histogram(image)

# ======================================================================================================================
        descriptors = image_database_app.sift_img_descriptor(image, sift_bovw="sift_bovw")
        sift_bovw = image_database_app.build_histogram(descriptors, kmeans_model)  # build histogram for each image

        print(sift_bovw)
# ======================================================================================================================


# ======================================================================================================================
# Insert data into the database
# ======================================================================================================================
        if img_vec is not None and img_hist is not None and sift_bovw is not None:  # check if an error occurred during vectorization of image (SIFT descriptor)
            img_vec = list(img_vec)
            print(len(img_vec))

            img_hist = list(img_hist)
            print(len(img_hist))

            sift_bovw = list(sift_bovw)
            print(len(sift_bovw))

            try:
                # json.dumps: stores image vector as string-text for better performance
                img_vec = json.dumps(img_vec)
                img_hist = json.dumps(img_hist)
                sift_bovw = json.dumps(sift_bovw)

                record_to_insert = (img_vec, input_path, img_hist, sift_bovw)
                cursor.execute(postgres_insert_query, record_to_insert)

                connection.commit()
                count = cursor.rowcount
                print(count, "Record inserted successfully into images table")

            except (Exception, psycopg2.Error) as error:
                if connection:
                    print("Failed to insert record into images table:", error)

    try:
        # ======================================================================================================================
        postgres_insert_kmeans_query = """INSERT INTO sift_bovw_clustering_model (kmeans_model) VALUES (%s)"""

        print(pickle.dumps(kmeans_model))
        # pickle.dumps: stores trained kmeans model
        model_to_insert = (psycopg2.Binary(pickle.dumps(kmeans_model)),)  # comma (,) is needed even though passing just one argument, can be replaced with [] with no comma
        print(model_to_insert)
        # psycopg2.Binary(

        cursor.execute(postgres_insert_kmeans_query, model_to_insert)
        # ======================================================================================================================

        connection.commit()
        count = cursor.rowcount
        print(count, "Record inserted successfully into sift_bovw_clustering_model table")

    except (Exception, psycopg2.Error) as error:
        if connection:
            print("Failed to insert record into sift_bovw_clustering_model table:", error)
    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")




vectorize_images()
