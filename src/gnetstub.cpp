#include "darknet.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <yolo_v2_class.hpp>

//"cfg/coco.data"

#define GGLIBRARY_API extern "C" __declspec(dllexport)

class ggNetInfo {
public:
    network net;
    char** names;
    list* options;
    ggNetInfo(char* datacfg = "cfg/coco.data", char* cfgfile = "cfg/yolov3.cfg", char* weightfile = "yolov3.weights",
        int benchmark_layers = 0) {
        options = read_data_cfg(datacfg);
        char* name_list = option_find_str(options, "names", "data/names.list");
        int names_size = 0;
        names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

        net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
        if (weightfile) {
            load_weights(&net, weightfile);
        }
        net.benchmark_layers = benchmark_layers;
        fuse_conv_batchnorm(net);
        calculate_binary_weights(net);
        if (net.layers[net.n - 1].classes != names_size) {
            printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
                name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
            //if (net.layers[net.n - 1].classes > names_size) getchar();
        }
        srand(2222222);
    }

    ~ggNetInfo() {
        free_ptrs((void**)names, net.layers[net.n - 1].classes);
        printf("names freed\n");
        free_list_contents_kvp(options);
        printf("free_list_contents_kvp freed\n");
        free_list(options);
        printf("options freed\n");
        free_network(net);
        printf("network freed");
    }
};
GGLIBRARY_API ggNetInfo* ggCreateNetwork(char* datacfg = "cfg/coco.data", char* cfgfile = "cfg/yolov3.cfg", char* weightfile = "yolov3.weights",
    int benchmark_layers = 0)
{
    ggNetInfo* info = new ggNetInfo(datacfg, cfgfile, weightfile, benchmark_layers);
   
    //free_network(net);
    return info;
}

GGLIBRARY_API void gFreeNetwork(ggNetInfo* net) {
    delete net;
}

char* getDetectionJson(detection* dets, int num, float thresh, char** names, int classes);
extern "C" image mat_to_image(cv::Mat mat);
GGLIBRARY_API void gDetect(ggNetInfo* info, char* imageBuffer, int imgLength, void(*action)(char*), float thresh = 0.24, float hier_thresh = 0.5f, int letter_box = 0) {
    network net = info->net;
    float nms = .45;    // 0.4F
    std::vector<char> imageDataVec = std::vector<char>(imageBuffer, imageBuffer + imgLength);
    cv::Mat imgMat = cv::imdecode(imageDataVec, -1);
    image im = mat_to_image(imgMat);
    //image im = load_image(input, 0, 0, net.c);
    image sized;
    if (letter_box) sized = letterbox_image(im, net.w, net.h);
    else sized = resize_image(im, net.w, net.h);
    layer l = net.layers[net.n - 1];

    //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
    //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)xcalloc(l.classes, sizeof(float));

    float* X = sized.data;

    //time= what_time_is_it_now();
    double time = get_time_point();
    network_predict(net, X);
    //network_predict_image(&net, im); letterbox = 1;
    //printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);
    //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));

    int nboxes = 0;
    detection* dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
    if (nms) {
        if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
        else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
    }

    //if (json_file) {
    long long json_image_id = 0;
    char * json_buf = getDetectionJson(dets, nboxes, thresh, info->names, l.classes);
    action(json_buf);
    //fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
    free(json_buf);
    //}
}

extern "C" int compare_by_probs(const void* a_ptr, const void* b_ptr);
char * getDetectionJson(detection* dets, int num, float thresh, char** names, int classes)
{
    int selected_detections_num;
    detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num, names);

      

    char* send_buf = (char*)calloc(1024, sizeof(char));
    if (!send_buf) return NULL;
    sprintf(send_buf, "[ \n");

    const int BUFMAX = 4096;
    char buf[BUFMAX];
    char lblBuf[BUFMAX];
    char probBuf[BUFMAX];
    char tmpNumBuf[256];
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
    for (int i = 0; i < selected_detections_num; ++i) {
        detection_with_class* detWithClass = selected_detections + i;
        int bestClass = detWithClass->best_class;
        detection* det = &detWithClass->det;
        box b = det->bbox;

        sprintf(buf, "  %s{\"class_id\":%d, \"name\":\"%s\", \"box\":{\"x\":%f, \"y\":%f, \"w\":%f, \"h\":%f}, \"confidence\":%f",
            i == 0?"":",\n",
            bestClass, names[bestClass], b.x, b.y, b.w, b.h, det->prob[bestClass]);

        strcpy(lblBuf, ",\"labels\":[");
        strcpy(probBuf, ",\"probs\":[");
        int lblBufLen = strlen(lblBuf)+1;
        int probBufLen = strlen(probBuf)+1;
        bool hasLblProb = false;
        for (int j = 0; j < classes; j++) {
            float probj = det->prob[j];
            if (probj > thresh && j != bestClass) {
                char* name = names[bestClass];
                int nameLen = strlen(name) + 3;
                sprintf(tmpNumBuf, "%f", probj);
                int probLen = strlen(tmpNumBuf) + 1;
                lblBufLen += nameLen;
                probBufLen += probLen;
                if (lblBufLen < BUFMAX && probBufLen < BUFMAX) {
                    if (hasLblProb) {
                        strcat(lblBuf, ",");
                        strcat(probBuf, ",");
                    }
                    strcat(lblBuf, "\"");
                    strcat(lblBuf, name);
                    strcat(lblBuf, "\"");
                    strcat(probBuf, tmpNumBuf);
                    hasLblProb = true;
                }
                else {
                    break;
                }                
            }
        }
        strcat(lblBuf, "]");
        strcat(probBuf, "]");
        int send_buf_len = strlen(send_buf);
        int buf_len = strlen(buf) + strlen(lblBuf)+ strlen(probBuf)+2;
        int total_len = send_buf_len + buf_len + 100;
        send_buf = (char*)realloc(send_buf, total_len * sizeof(char));
        if (!send_buf) {
            if (buf) free(buf);
            return 0;// exit(-1);
        }
        strcat(send_buf, buf);
        strcat(send_buf, lblBuf);
        strcat(send_buf, probBuf);
        strcat(send_buf, "}");
    }
    free(selected_detections);
    
    strcat(send_buf, "]");
    return send_buf;
}


void test_detector1(char* datacfg = "cfg/coco.data", char* cfgfile = "cfg/yolov3.cfg", char* weightfile = "yolov3.weights", char* filename = NULL, float thresh = 0.24,
    float hier_thresh = 0.5f, int dont_show = 0, int ext_output = 0, int save_labels = 0, char* outfile = NULL, int letter_box = 0, int benchmark_layers = 0)
{
    list* options = read_data_cfg(datacfg);
    char* name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char** names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image** alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }
    srand(2222222);
    char buff[256];
    char* input = buff;
    char* json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        if (!json_file) {
            error("fopen failed");
        }
        char* tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if (letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n - 1];

        //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
        //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)xcalloc(l.classes, sizeof(float));

        float* X = sized.data;

        //time= what_time_is_it_now();
        double time = get_time_point();
        network_predict(net, X);
        //network_predict_image(&net, im); letterbox = 1;
        printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);
        //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));

        int nboxes = 0;
        detection* dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
        save_image(im, "predictions");
        if (!dont_show) {
            show_image(im, "predictions");
        }

        if (json_file) {
            if (json_buf) {
                char* tmp = ", \n";
                fwrite(tmp, sizeof(char), strlen(tmp), json_file);
            }
            ++json_image_id;
            json_buf = detection_to_json(dets, nboxes, l.classes, names, json_image_id, input);

            fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
            free(json_buf);
        }

        // pseudo labeling concept - fast.ai
        if (save_labels)
        {
            char labelpath[4096];
            replace_image_to_label(input, labelpath);

            FILE* fw = fopen(labelpath, "wb");
            int i;
            for (i = 0; i < nboxes; ++i) {
                char buff[1024];
                int class_id = -1;
                float prob = 0;
                for (j = 0; j < l.classes; ++j) {
                    if (dets[i].prob[j] > thresh&& dets[i].prob[j] > prob) {
                        prob = dets[i].prob[j];
                        class_id = j;
                    }
                }
                if (class_id >= 0) {
                    sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f\n", class_id, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
                    fwrite(buff, sizeof(char), strlen(buff), fw);
                }
            }
            fclose(fw);
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        if (filename) break;
    }

    if (json_file) {
        char* tmp = "\n]";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
        fclose(json_file);
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}
