enum FilterType { FILTER_BLUR, FILTER_EDGE, FILTER_EMBOSS, FILTER_SEPIA, FILTER_INVERT, FILTER_GREY, FILTER_ORTON };
int filter_radius = 1;
int filter_type = FILTER_BLUR;
pthread_mutex_t filter_lock = PTHREAD_MUTEX_INITIALIZER;
int update_requested = 1;

void generate_blur_filter(float *filter, int radius) {
    int size = 2 * radius + 1;
    float val = 1.0f / (size * size);
    for (int i = 0; i < size * size; ++i) filter[i] = val;
    
}

void generate_edge_filter(float *f) {
    float edge[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    memcpy(f, edge, sizeof(float) * 9);
}

void generate_emboss_filter(float *f) {
    float emboss[9] = {
        -2, -1, 0,
        -1,  1, 1,
         0,  1, 2
    };
    memcpy(f, emboss, sizeof(float) * 9);
}

void choose_filter(float *h_filter, int r, int type){

    if (type == FILTER_BLUR || type == FILTER_ORTON) generate_blur_filter(h_filter, r);
    
    else if (type == FILTER_EDGE || type == FILTER_EMBOSS) {

        h_filter = (float *)realloc(h_filter, sizeof(float) * 9);
        if (type == FILTER_EDGE) generate_edge_filter(h_filter);
        else generate_emboss_filter(h_filter);
    }
}

void *cli_thread(void *arg) {
    char cmd[256];
    while (1) {
        
        if (fgets(cmd, sizeof(cmd), stdin)) {

            pthread_mutex_lock(&filter_lock);
            int r;
            if (sscanf(cmd, "%d", &r) == 1 && r >= 0 && r <= 10) {
                
                filter_radius = r;
                filter_type = FILTER_BLUR;
                update_requested = 1;
                
            } 
            else if (strncmp(cmd, "edge", 4) == 0) {
                filter_type = FILTER_EDGE;
                filter_radius = 1;
                update_requested = 1;
            }
            else if (strncmp(cmd, "emboss", 6) == 0) {
                filter_type = FILTER_EMBOSS;
                filter_radius = 1;
                update_requested = 1;
            }
            else if (strncmp(cmd, "invert", 6) == 0) {
                filter_type = FILTER_INVERT;
                filter_radius = 1;
                update_requested = 1;
            }
            else if (strncmp(cmd, "sepia", 5) == 0) {
                filter_type = FILTER_SEPIA;
                filter_radius = 1;
                update_requested = 1;
            }
            else if (strncmp(cmd, "grey", 4) == 0) {
                filter_type = FILTER_GREY;
                filter_radius = 1;
                update_requested = 1;
            }
            else if (strncmp(cmd, "orton", 5) == 0) {
                filter_type = FILTER_ORTON;
                filter_radius = 1;
                update_requested = 1;
            }
            else if (strncmp(cmd, "exit", 4) == 0) update_requested = 2;

            else printf("Invalid");
            pthread_mutex_unlock(&filter_lock);
        }
    }
    return NULL;
}