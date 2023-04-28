# transfer different annotation format of object detection dataset

# yolo format: <object-class> <x_center> <y_center> <width> <height> (normalized by image width and height)
# coco format: <object-class> <x_top_left> <y_top_left> <width> <height> 
# voc format: <object-class> <x_top_left> <y_top_left> <x_bottom_right> <y_bottom_right>
# 
from pathlib import Path
import shutil
from tqdm import tqdm

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

import json
import argparse

image_suffixes = ['.jpg', '.jpeg', '.png', '.bmp']

class DetectionAnnotationTransferer:
    def __init__(self):
        '''
            Annotation transferer for object detection dataset

            Support yolo, coco, voc format
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
                        
        '''
        self.classes = [] # list of category names(str)
        self.images = [] # list of image info(dict) {'path': image_path, 'hw': image_hw(tuple) (height, width)}
        self.bboxes = [] # list of bbox(dict) {'image_id': image_id(int), 'category_id': category_id(int), 'bbox': bbox(tuple) (x1, y1, x2, y2)}
    def addCategory(self, categoryName):
        '''
            add category name to self.classes
            duplicate category name is not allowed(raise ValueError)
            
            example: self.classes = ['dog']
            >>>self.addCategory('cat')
            self.classes = ['dog', 'cat']
            
            Args:
                categoryName: str
        '''
        if categoryName not in self.classes:
            self.classes.append(categoryName)
        else:
            raise ValueError('category already exists')

    def addImage(self, image_path, image_hw):
        '''
            add image info to self.images
            
            example: self.images = [{'path': 'path/to/image', 'hw': (height, width)}]
            >>>self.addImage('path/to/image2', (height2, width2))
            self.images = [{'path': 'path/to/image', 'hw': (height, width)}, {'path': 'path/to/image2', 'hw': (height2, width2)}]
            
            Args:
                image_path: Path
                image_hw: tuple (height, width) (int, int)
            

        '''
        self.images.append({'path': image_path, 'hw': image_hw})

    def addBboxes(self, image_id, category_id, bbox):
        '''
            add bbox info to self.bboxes
            bbox coordinate is absolute coordinate whether int or float

            example: self.bboxes = [{'image_id': 0, 'category_id': 0, 'bbox': (x1, y1, x2, y2)}]
            >>>self.addBboxes(0, 1, (x1, y1, x2, y2))
            self.bboxes = [{'image_id': 0, 'category_id': 0, 'bbox': (x1, y1, x2, y2)}, {'image_id': 0, 'category_id': 1, 'bbox': (x1, y1, x2, y2)}]
            
            Args:
                image_id: int
                category_id: int
                bbox: tuple (x1, y1, x2, y2) (float, float, float, float)
        '''
        self.bboxes.append({'image_id': image_id, 'category_id': category_id, 'bbox': bbox})

    def load(self, format, annotationPath:Path):
        if format == 'yolo':
            self.loadYolo(annotationPath)
        elif format == 'coco':
            self.loadCoco(annotationPath)
        elif format == 'voc':
            self.loadVoc(annotationPath)
        else:
            print(format)
            raise ValueError('unsupported annotation format: ' + format)
    def dump(self, format, annotationPath:Path):
        if format == 'yolo':
            self.dumpYolo(annotationPath)
        elif format == 'coco':
            self.dumpCoco(annotationPath)
        elif format == 'voc':
            self.dumpVoc(annotationPath)
        else:
            raise ValueError('unsupported annotation format: ' + format)

    def loadYolo(self, datasetPath:Path):
        '''
            load yolo format annotation from datasetPath along with images path and height and width
            

            datasetPath should contain 'images' folder and 'labels' folder and 'classes.txt' file

            'images' folder contains images e.g. 'images/0001.jpg'
            'labels' folder contains corresponding label files e.g. 'labels/0001.txt'
            'classes.txt' contains category names in each line e.g. 'dog\\n' 'cat\\n'
        '''
        imageFolderPath = datasetPath / 'images'
        labelFolderPath = datasetPath / 'labels'

        with open(datasetPath / 'classes.txt', 'r') as f:
            classes = f.read().strip().split('\n')
        for className in classes:
            self.addCategory(className)
        
        for path in tqdm(imageFolderPath.iterdir(), desc="loading YOLO format..."):
            if path.suffix.lower() not in image_suffixes:
                raise TypeError('unsupported image format: ' + str(path))
            # read image by cv2 supporting chinese path
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)       
            int_height, int_width = image.shape[:2]
            
            # add image info
            self.addImage(path.absolute(), (int_height, int_width))

            # read corresponding label file
            correspondingLabelPath = labelFolderPath / (path.stem + '.txt')
            if not correspondingLabelPath.exists():
                raise FileNotFoundError('corresponding label file not found: ' + str(correspondingLabelPath))
            with open(correspondingLabelPath, 'r') as f:
                lines = f.read().strip().split('\n')
            for line in lines:
                line = line.strip().split()
                if len(line) != 5:
                    raise ValueError('invalid label format: ' + str(correspondingLabelPath))
                category_id = int(line[0])
                # bbox = [x_center, y_center, width, height] (normalized by image width and height)
                xc, yc, w, h = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]

                # convert to [x1, y1, x2, y2]
                x1, y1, x2, y2 = xc - w/2, yc - h/2, xc + w/2, yc + h/2
                x1, y1, x2, y2 = x1 * int_width, y1 * int_height, x2 * int_width, y2 * int_height

                # add bbox info 
                # retain coordinate not rounded and int for further processing
                self.addBboxes(len(self.images)-1, category_id, [x1, y1, x2, y2]) # image_id, category_id, bbox
    def loadCoco(self, datasetPath:Path):
        '''
            load coco format annotation from datasetPath along with images path and height and width

            datasetPath should contain 'images' folder 'coco.json' file

            'images' folder contains images e.g. 'images/0001.jpg'
            'coco.json' contains coco dataset in json format

        '''
        with open(datasetPath / "coco.json", 'r') as f:
            # load coco dataset in json format
            cocoDataset = json.load(f)
            # sort by id
            cocoDataset['categories'].sort(key=lambda x: x['id'])
            cocoDataset['images'].sort(key=lambda x: x['id'])
            cocoDataset['annotations'].sort(key=lambda x: x['image_id'])
            # add categories
            for category in cocoDataset['categories']:
                self.addCategory(category['name'])
            # add images
            for image in cocoDataset['images']:
                self.addImage(Path(image['file_name']).absolute(), (image['height'], image['width']))
            # add bboxes
            for annotation in tqdm(cocoDataset['annotations'], desc="loading COCO format..."):
                category_id = annotation['category_id']
                bbox = annotation['bbox'] # [x_top_left, y_top_left, width, height]
                # convert to [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3] # x1, y1, x2, y2

                # add bbox info
                # retain coordinate not rounded and int for further processing
                self.addBboxes(annotation['image_id'], category_id, [x1, y1, x2, y2])
    def loadVoc(self, datasetPath:Path):
        '''
            load voc format annotation from datasetPath along with images path and height and width

            datasetPath should contain 'JPEGImages', folder 'Annotations' folder and 'classes.txt' file

            'JPEGImages' folder contains images e.g. 'JPEGImages/0001.jpg'
            'Annotations' folder contains corresponding annotation files e.g. 'Annotations/0001.xml'
            'classes.txt' contains category names in each line e.g. 'dog\\n' 'cat\\n'

        '''
        imageFolderPath = datasetPath / 'JPEGImages'
        annotationFolderPath = datasetPath / 'Annotations'
        with open(datasetPath / 'classes.txt', 'r') as f:
            classes = f.read().strip().split('\n')
        for className in classes:
            self.addCategory(className)

        for path in tqdm(imageFolderPath.iterdir(), desc="loading VOC format..."):
            if path.suffix.lower() not in image_suffixes:
                raise TypeError('unsupported image format: ' + str(path))
            # read image by cv2 supporting chinese path
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)       
            int_height, int_width = image.shape[:2]
            
            # add image info
            self.addImage(path.absolute(), (int_height, int_width))

            # read corresponding annotation file
            correspondingAnnotationPath = annotationFolderPath / (path.stem + '.xml')
            if not correspondingAnnotationPath.exists():
                raise FileNotFoundError('corresponding annotation file not found: ' + str(correspondingAnnotationPath))
            tree = ET.parse(correspondingAnnotationPath)
            root = tree.getroot()
            for obj in root.iter('object'):
                category_id = self.classes.index(obj.find('name').text)
                bbox = obj.find('bndbox')
                x1, y1, x2, y2 = float(bbox.find('xmin').text), float(bbox.find('ymin').text), float(bbox.find('xmax').text), float(bbox.find('ymax').text)
                # add bbox info
                self.addBboxes(len(self.images)-1, category_id, [x1, y1, x2, y2])
    def dumpYolo(self, datasetPath:Path):
        '''
            dump yolo format annotation to datasetPath

            images will be copied to datasetPath / 'images' folder
            labels will be dumped to datasetPath / 'labels' folder
            classes will be dumped to datasetPath / 'classes.txt' file

            datasetPath will contain 'images' folder 'labels' folder and 'classes.txt' file

            'images' folder contains images e.g. 'images/0001.jpg'
            'labels' folder contains corresponding annotation files e.g. 'labels/0001.txt'
            'classes.txt' contains category names in each line e.g. 'dog\\n' 'cat\\n'
            
        '''
        imageFolderPath = datasetPath / 'images'
        labelFolderPath = datasetPath / 'labels'
        imageFolderPath.mkdir(parents=True, exist_ok=True)
        labelFolderPath.mkdir(parents=True, exist_ok=True)
        with open(datasetPath / 'classes.txt', 'w') as f:
            f.write('\n'.join(self.classes))

        for image_id, image in tqdm(enumerate(self.images), desc="dumping YOLO format..."):
            image_path = image['path']
            height, width = height, width = image['hw']
            shutil.copy(str(image_path), str(imageFolderPath / image_path.name))
            correspondingLabels = [ bbox for bbox in self.bboxes if bbox['image_id'] == image_id]
            with open(labelFolderPath / (image_path.stem + '.txt'), 'w') as f:
                for bbox in correspondingLabels:
                    category_id = bbox['category_id']
                    x1, y1, x2, y2 = bbox['bbox']
                    # convert to [x_center, y_center, width, height] (normalized by image width and height)
                    xc, yc, w, h = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height
                    f.write('%d %.6f %.6f %.6f %.6f\n' % (category_id, xc, yc, w, h))

    
    def dumpCoco(self, datasetPath:Path):
        '''
            dump coco format annotation to datasetPath

            images will be copied to datasetPath / 'images' folder
            labels will be dumped to datasetPath / 'coco.json' file

            datasetPath will contain 'images' folder and 'coco.json' file


        '''

        imageFolderPath = datasetPath / 'images'
        imageFolderPath.mkdir(parents=True, exist_ok=True)
        
        # coco format dataset
        cocoDataset = {'categories':[], 'annotations':[], 'images':[]}
        for i, className in enumerate(self.classes):
            cocoDataset['categories'].append({'id': i, 'name': className, 'supercategory': 'mark'})
        for image_id, image in tqdm(enumerate(self.images), desc="dumping COCO format..."):
            image_path = image['path']
            height, width = image['hw']
            shutil.copy(str(image_path), str(imageFolderPath / image_path.name))
            # add image info
            cocoDataset['images'].append({'id': image_id, 'file_name': str((imageFolderPath / image_path.name).absolute()), 'height': height, 'width': width})
            correspondingLabels = [ (annotationId, bbox) for (annotationId, bbox) in enumerate(self.bboxes) if bbox['image_id'] == image_id]
            for (annotationId, bbox) in correspondingLabels:
                category_id = bbox['category_id']
                x1, y1, x2, y2 = bbox['bbox']
                x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                # convert to [x_top_left, y_top_left, width, height]
                x_top_left, y_top_left, w, h = x1, y1, x2 - x1, y2 - y1
                # add bbox info
                cocoDataset['annotations'].append({'id': annotationId, 'image_id': image_id, 'category_id': category_id, 'bbox': [x_top_left, y_top_left, w, h], 'area': width * height, 'iscrowd': 0, 'segmentation': [x1, y1, x2, y1, x2, y2, x1, y2]})
            
        with open(datasetPath / 'coco.json', 'w') as f:
            json.dump(cocoDataset, f, indent=2)
    
    def dumpVoc(self, datasetPath:Path):
        '''
            dump voc format annotation to datasetPath
            
            images will be copied to datasetPath / 'JPEGImages' folder
            labels will be dumped to datasetPath / 'Annotations' folder
            classes will be dumped to datasetPath / 'classes.txt' file

            datasetPath will contain 'JPEGImages' folder 'Annotations' folder and 'classes.txt' file
        '''

        imageFolderPath = datasetPath / 'JPEGImages'
        annotationFolderPath = datasetPath / 'Annotations'
        imageFolderPath.mkdir(parents=True, exist_ok=True)
        annotationFolderPath.mkdir(parents=True, exist_ok=True)
        with open(datasetPath / 'classes.txt', 'w') as f:
            f.write('\n'.join(self.classes))
        for image_id, image in tqdm(enumerate(self.images), desc="dumping VOC format..."):
            image_path = image['path']
            height, width = image['hw']
            shutil.copy(str(image_path), str(imageFolderPath / image_path.name))
            # filter corresponding bboxes
            correspondingLabels = [ (annotationId, bbox) for (annotationId, bbox) in enumerate(self.bboxes) if bbox['image_id'] == image_id]
            # write corresponding annotation file into xml
            root = ET.Element('annotation') # create root node
            ET.SubElement(root, 'folder').text = 'JPEGImages'
            ET.SubElement(root, 'filename').text = image_path.name
            ET.SubElement(root, 'path').text = str((imageFolderPath / image_path.name).absolute())
            source = ET.SubElement(root, 'source')
            ET.SubElement(source, 'database').text = 'Unknown' 
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(width)
            ET.SubElement(size, 'height').text = str(height)
            ET.SubElement(size, 'depth').text = '3' # TODO: support grayscale image
            ET.SubElement(root, 'segmented').text = '0' 

            # write corresponding bboxes annotation into xml
            for (annotationId, bbox) in correspondingLabels:
                category_id = bbox['category_id']
                x1, y1, x2, y2 = bbox['bbox']
                x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                # add bbox info
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = self.classes[category_id]
                ET.SubElement(obj, 'pose').text = 'Unspecified' 
                ET.SubElement(obj, 'truncated').text = '0' 
                ET.SubElement(obj, 'difficult').text = '0'
                bbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bbox, 'xmin').text = str(x1)
                ET.SubElement(bbox, 'ymin').text = str(y1)
                ET.SubElement(bbox, 'xmax').text = str(x2)
                ET.SubElement(bbox, 'ymax').text = str(y2)
            # write pretty-printed xml into file
            tree = ET.ElementTree(root)
            tree.write(str(annotationFolderPath / (image_path.stem + '.xml')), encoding='utf-8')
            dom = minidom.parse(str(annotationFolderPath / (image_path.stem + '.xml')))
            with open(str(annotationFolderPath / (image_path.stem + '.xml')), 'w') as f:
                f.write(dom.toprettyxml(indent='\t'))

# a parser for command line arguments
# Arguments:
# -i: input dataset path
# -o: output dataset path
# -t: input format, support 'yolo', 'coco', 'voc'
# -f: output format, support 'yolo', 'coco', 'voc'
def getParser():
    parser = argparse.ArgumentParser(description='''            Annotation transferer for object detection dataset

            Support yolo, coco, voc format
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
            'classes.txt' contains category names in each line e.g. 'dog\\n' 'cat\\n''')
    parser.add_argument('-i', '--input', type=str, help='input dataset path')
    parser.add_argument('-o', '--output', type=str, help='output dataset path')
    parser.add_argument('-t', '--input-format', type=str, help='input dataset format, support yolo, coco, voc')
    parser.add_argument('-f', '--output-format', type=str, help='output dataset format, support yolo, coco, voc')
    return parser

if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    inputPath = Path(args.input)
    outputPath = Path(args.output)
    inputFormat = args.input_format.lower()
    outputFormat = args.output_format.lower()
    assert inputFormat in ['yolo', 'coco', 'voc'], 'input format not supported'
    assert outputFormat in ['yolo', 'coco', 'voc'], 'output format not supported'
    assert inputPath.exists(), 'input path not exists'
    assert inputPath.is_dir(), 'input path is not a directory'

    assert inputPath != outputPath, 'input path and output path should not be the same'

    transferer = DetectionAnnotationTransferer()
    transferer.load(inputFormat, inputPath)
    transferer.dump(outputFormat, outputPath)
    print('done')