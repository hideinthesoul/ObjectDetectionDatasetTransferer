# ObjectDetectionToolScripts
Tools for dataset format transferring, etc.

Annotation transferer for object detection dataset
# Arguments:
-i: input dataset path
-o: output dataset path
-t: input format, support 'yolo', 'coco', 'voc'
-f: output format, support 'yolo', 'coco', 'voc'

# Support yolo, coco, voc format
            ----------------------------------------------------------------------------------------

            load yolo format annotation from datasetPath along with images path and height and width
            

            datasetPath should contain 'images' folder and 'labels' folder and 'classes.txt' file

            'images' folder contains images e.g. 'images/0001.jpg'
            'labels' folder contains corresponding label files e.g. 'labels/0001.txt'
            'classes.txt' contains category names in each line e.g. 'dog\\n' 'cat\\n'
            ----------------------------------------------------------------------------------------
            
            load coco format annotation from datasetPath along with images path and height and width

            datasetPath should contain 'images' folder 'coco.json' file

            'images' folder contains images e.g. 'images/0001.jpg'
            'coco.json' contains coco dataset in json format
            ----------------------------------------------------------------------------------------
            load voc format annotation from datasetPath along with images path and height and width

            datasetPath should contain 'JPEGImages', folder 'Annotations' folder and 'classes.txt' file

            'JPEGImages' folder contains images e.g. 'JPEGImages/0001.jpg'
            'Annotations' folder contains corresponding annotation files e.g. 'Annotations/0001.xml'
            'classes.txt' contains category names in each line e.g. 'dog\\n' 'cat\\n'