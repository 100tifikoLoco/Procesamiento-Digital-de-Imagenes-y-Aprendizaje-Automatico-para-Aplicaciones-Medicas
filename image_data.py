import pandas as pd
from progressbar import ProgressBar
import cv2
import os

def extract_images(video_path, image_path, cropped=False, max_frames=None, target_class=['COVID', 'Pneumonia', 'Normal'], target_source=['Butterfly', 'GrepMed', 'LITFL', 'PocusAtlas'], target_probe=['convex', 'linear']):#
    print("Realizando extracción de imagenes")
    """
        Función para extraer imágenes de archivos de video de ultrasonido.
    Parámetros:
        - video_path: Ruta de la carpeta de video para leer archivos de video (carpeta de origen)
        - image_path: Ruta de la carpeta de imágenes para almacenar las imágenes extraídas en ella 
                    (carpeta de destino)
        - cropped: Si es True, los fotogramas se extraerán de los archivos de video recortados. 
                    De lo contrario, se trabajara con los archivos de video originales.
        - max_frames: Número máximo de fotogramas (imágenes) que se extraerán de un archivo de video. 
                    Nota: si un archivo de video tiene menos fotogramas que el max_frames solicitado, 
                    se extraerán todos los fotogramas
        - target_class: Las clases de destino para las que el usuario desea extraer imágenes
        - target_source: Las fuentes de datos de destino para extraer imágenes
        - target_probe: Filtro para identificar el tipo de sonda para la que se extraerán imágenes
        org_framecount
    """
    try:
        if cropped: #En caso de ser archivos recortados
            # Lee los metadatos de videos recortados
            vid_prop_df = pd.read_csv('utils/video_cropping_metadata.csv', sep=',', encoding='latin1')
        else: 
            # Lee los metadatos de archivos de video 
            metadata = pd.read_csv('utils/video_metadata.csv', sep=',', encoding='latin1')
            metadata = metadata[metadata.id !='22_butterfly_covid'] # Se omite el archivo #22 de butterfly 

            # Lee el archivo de propiedades de los videos 
            vid_prop_df = pd.read_csv('utils/video_files_properties.csv')
            vid_prop_df = vid_prop_df[vid_prop_df.filename !='22_butterfly_covid.mp4'] # Se omite el archivo #22 de butterfly 

            # Fusiona con el archivo de metadatos de video
            vid_prop_df.filename = vid_prop_df.filename.astype(str)
            vid_prop_df.filename = vid_prop_df.filename.str.strip()

            metadata['filename2'] = metadata.id + '.' + metadata.filetype
            metadata.filename2 = metadata.filename2.astype(str)
            metadata.filename2 = metadata.filename2.str.strip()

            vid_prop_df = pd.merge(vid_prop_df, metadata[['filename2', 'source', 'probe', 'class']], left_on='filename', right_on='filename2', how='left').drop('filename2', axis=1)
            del metadata['filename2']

        # Extrae imagenes en función de los parametros dados  
        progress = ProgressBar(max_value=vid_prop_df.shape[0])
    
        for idx, row in progress(vid_prop_df.iterrows()): # Libera la condicion de corte en el marco de datos tras realizar la prueba 
            if cropped:
                filename = row.filename.split('.')[0] + '_prc.avi'
                #Omitir archivo 160 por errores en el recorte
                if filename == '160_core_other_prc.avi':
                    print(f"Omitiendo el archivo recortado {filename}")
                    continue
                file_id = filename.split('.')[0]
                frame_count = row.org_framecount
            else:
                filename = row.filename
                file_id = row.filename.split('.')[0]
                #frame_rate = row.framerate
                frame_count = row.frame_count
        
            vid_probe = row.probe.lower()

            # Lee el archivo de video y configura la extracción de fotogramas
            try:
                cv2video = cv2.VideoCapture(video_path + str(filename))

                if max_frames:
                    img_pos = int(frame_count / max_frames)
                    n_frames = 1

                while cv2video.isOpened(): 
                    frame_id = cv2video.get(1)  #Numero de cuadro actual
                    ret, frame = cv2video.read()
                    if (ret != True):
                        break
                    
                    # Guarda el fotograma si cumple las condiciones
                    if (max_frames) and (img_pos) and (frame_id % img_pos == 0) and (n_frames <= max_frames): # and (frame_count > max_frames):
                        img_filename = os.path.join(image_path, file_id + "_" + vid_probe + "_frame%d.jpg" % frame_id)
                        if max_frames:
                            n_frames += 1
                    else:
                        img_filename = os.path.join(image_path, file_id + "_" + vid_probe + "_frame%d.jpg" % frame_id)
                    cv2.imwrite(img_filename, frame)
                cv2video.release()    
            except Exception as e:
                print(f"Ocurrio un error intentando procesar el archivo de video {filename}: {e}")
    except Exception as e:
        print(f"Error general durante la extracción de imagenes: {e}")
        
"""
    if cropped:
        # read cropped videos metadata file
        vid_prop_df = pd.read_csv('utils/video_cropping_metadata.csv', sep=',', encoding='latin1')
    else:
        # read videos metadata file
        metadata = pd.read_csv('utils/video_metadata.csv', sep=',', encoding='latin1')
        metadata = metadata[metadata.id !='22_butterfly_covid'] # 22_butterfly_covid.mp4 was removed in March release of butterfly


        # read videos' properties file
        vid_prop_df = pd.read_csv('utils/video_files_properties.csv')
        vid_prop_df = vid_prop_df[vid_prop_df.filename !='22_butterfly_covid.mp4'] # 22_butterfly_covid.mp4 was removed in March release of butterfly

        # merge with the video meta data file 
        vid_prop_df.filename = vid_prop_df.filename.astype(str)
        vid_prop_df.filename = vid_prop_df.filename.str.strip()

        metadata['filename2'] = metadata.id + '.' + metadata.filetype
        metadata.filename2 = metadata.filename2.astype(str)
        metadata.filename2 = metadata.filename2.str.strip()

        vid_prop_df = pd.merge(vid_prop_df, metadata[['filename2', 'source', 'probe', 'class']], left_on='filename', right_on='filename2', how='left').drop('filename2', axis=1)
        del metadata['filename2']

    # extract images based on the given parameters    
    progress = ProgressBar(max_value=vid_prop_df.shape[0])
    for idx, row in progress(vid_prop_df.iterrows()): # reselase the slicing condition on the dataframe after test is done
        if cropped:
            filename = row.filename.split('.')[0] + '_prc.avi'
            file_id = filename.split('.')[0]
            frame_count = row.org_framecount
        else:
            filename = row.filename
            file_id = row.filename.split('.')[0]
            #frame_rate = row.framerate
            frame_count = row.frame_count
        
        vid_probe = row.probe.lower()

        # read the video file and extracting frames
        cv2video = cv2.VideoCapture(video_path + str(filename))

        if max_frames:
            img_pos = int(frame_count / max_frames)
            n_frames = 1
        
        while cv2video.isOpened(): 
            frame_id = cv2video.get(1)  #current frame number
            ret, frame = cv2video.read()
            if (ret != True):
                break
            
            # storing frames
            if (max_frames) and (img_pos): # and (frame_count > max_frames):
                if (frame_id % img_pos == 0) and (n_frames <= max_frames):
                    img_filename = os.path.join(image_path, file_id + "_" + vid_probe + "_frame%d.jpg" % frame_id)
                    n_frames += 1
            else:
                img_filename = os.path.join(image_path, file_id + "_" + vid_probe + "_frame%d.jpg" % frame_id)

            cv2.imwrite(img_filename, frame)
        cv2video.release()    
"""