#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//function which displays annotation into a JSON format and export it into a JSON file
void export_annotation(image im, int num, float thresh, box *boxes, float **probs, char **names, int classes, char *cfgfile)
{
	//array which will contain bounding box into a JSON format
	char ** bounding_boxes=calloc(num,sizeof(char *));
	int index = 0;

	for(int i = 0; i < num; ++i){

		int class = max_index(probs[i], classes);
		float prob = probs[i][class];

		if(prob > thresh){

			//we get the bounding box and its value (corner, width, etc.)
			box b = boxes[i];
			int left  = (b.x-b.w/2.)*im.w;
			int right = (b.x+b.w/2.)*im.w;
			int top   = (b.y-b.h/2.)*im.h;
			int bot   = (b.y+b.h/2.)*im.h;
			if(left < 0) left = 0;
			if(right > im.w-1) right = im.w-1;
			if(top < 0) top = 0;
			if(bot > im.h-1) bot = im.h-1;
			int width = right-left;
			int height = bot-top;

			bounding_boxes[index]=calloc(10000,sizeof(char));
			//we build the JSON annotation of the bounding box and add it into the bounding_boxes array
			sprintf(bounding_boxes[index],"{\"corner\":{\"x\":%d,\"y\":%d},\"width\":%d,\"height\":%d,\"class\":\"%s\",\"probability\":%f}",left,top,width,height,names[class],prob);
			index++;
		}

	}

	//we build a JSON array countaining the bounding boxes annotations
	char *bd_boxes = calloc(10000,sizeof(char));
	//if there is 0 bounding box
	if (index==0){
		sprintf(bd_boxes,"[]");
	}
	else{
		sprintf(bd_boxes,"[%s",bounding_boxes[0]);
		for(int i=1;i<index;i++){
			sprintf(bd_boxes,"%s,%s",bd_boxes,bounding_boxes[i]);
		}
		sprintf(bd_boxes,"%s]",bd_boxes);
	}
	//we build the annotator id
	char * annotator_id=calloc(100,sizeof(char));
	if(0==strcmp(cfgfile,"cfg/yolo.cfg")){
		printf("%s\n",cfgfile );
		strncpy(annotator_id,"mdl_YOLOv1_coco",15);
	}
	else if(0==strcmp(cfgfile,"cfg/yolo.2.0.cfg")){
		strncpy(annotator_id,"mdl_YOLOv2_coco",15);
	}
	else if(0==strcmp(cfgfile,"cfg/yolo-voc.cfg")){
		strncpy(annotator_id,"mdl_YOLOv1_voc",14);
	}
	else if(0==strcmp(cfgfile,"cfg/yolo-voc.2.0.cfg")){
		strncpy(annotator_id,"mdl_YOLOv2_voc",14);
	}
	else if(0==strcmp(cfgfile,"cfg/yolo9000.cfg")){
		strncpy(annotator_id,"mdl_YOLO9000",12);
	}
	else {
		strncpy(annotator_id,"mdl_YOLO",8);
	}
	//we create a JSON annotation for the image countaining the bounding boxes annotations, the image id and the annotator id
	char *annotation = calloc(10000,sizeof(char));
	sprintf(annotation,"{\"photo_id\":\"%s\",\"annotator_id\":\"%s\",\"bounding_boxes\":%s}",im.id,annotator_id,bd_boxes);
	//we print this annotation
	printf("Annotation :\n%s\n",annotation);

	//we export the annotation into a JSON file in the folder annotations/
	//we create the file name of out JSON file. It will be annotation_<image_id>.json
	char *json_filename = calloc(10000,sizeof(char));
	sprintf(json_filename,"annotations/annotation_%s_%s.json",im.id,annotator_id);
	FILE* json_file = fopen(json_filename,"w+");
	//we write the annotation into the JSON file
	fputs(annotation, json_file);
	fclose(json_file);

	//Free section
	for(int i=1;i<index;i++){
		free(bounding_boxes[i]);
	}
	free(bounding_boxes);
	free(bd_boxes);
	free(json_filename);
	free(annotation);
}
