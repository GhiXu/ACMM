#ifndef _ACMM_H_
#define _ACMM_H_

#include "main.h"

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);

Camera ReadCamera(const std::string &cam_path);
void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera);
float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);
void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc);

void RunJBU(const cv::Mat_<float>  &scaled_image_float, const cv::Mat_<float> &src_depthmap, const std::string &dense_folder , const Problem &problem);

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

void CudaSafeCall(const cudaError_t error, const std::string& file, const int line);
void CudaCheckError(const char* file, const int line);

struct cudaTextureObjects {
    cudaTextureObject_t images[MAX_IMAGES];
};

struct PatchMatchParams {
    int max_iterations = 3;
    int patch_size = 11;
    int num_images = 5;
    int max_image_size=3200;
    int radius_increment = 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int top_k = 4;
    float baseline = 0.54f;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    float disparity_min = 0.0f;
    float disparity_max = 1.0f;

    float scaled_cols;
    float scaled_rows;

    bool geom_consistency = false;
    bool multi_geometry = false;
    bool hierarchy = false;
    bool upsample = false;
};

class ACMM {
public:
    ACMM();
    ~ACMM();

    void InuputInitialization(const std::string &dense_folder, const std::vector<Problem> &problem, const int idx);
    void Colmap2MVS(const std::string &dense_folder, std::vector<Problem> &problems);
    void CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem);
    void RunPatchMatch();
    void SetGeomConsistencyParams(bool multi_geometry);
    void SetHierarchyParams();

    int GetReferenceImageWidth();
    int GetReferenceImageHeight();
    cv::Mat GetReferenceImage();
    float4 GetPlaneHypothesis(const int index);
    float GetCost(const int index);
private:
    int num_images;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> depths;
    std::vector<Camera> cameras;
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    float4 *plane_hypotheses_host;
    float4 *scaled_plane_hypotheses_host;
    float *costs_host;
    float *pre_costs_host;
    PatchMatchParams params;

    Camera *cameras_cuda;
    cudaArray *cuArray[MAX_IMAGES];
    cudaArray *cuDepthArray[MAX_IMAGES];
    cudaTextureObjects *texture_objects_cuda;
    cudaTextureObjects *texture_depths_cuda;
    float4 *plane_hypotheses_cuda;
    float4 *scaled_plane_hypotheses_cuda;
    float *costs_cuda;
    float *pre_costs_cuda;
    curandState *rand_states_cuda;
    unsigned int *selected_views_cuda;
    float *depths_cuda;
};

struct TexObj {
    cudaTextureObject_t imgs[MAX_IMAGES];
};

struct JBUParameters {
    int height;
    int width;
    int s_height;
    int s_width;
    int Imagescale;
};

struct JBUTexObj {
    cudaTextureObject_t imgs[JBU_NUM];
};

class JBU {
public:
    JBU();
    ~JBU();

    // Host Parameters
    float *depth_h;
    JBUTexObj jt_h;
    JBUParameters jp_h;

    // Device Parameters
    float *depth_d;
    cudaArray *cuArray[JBU_NUM]; // The first for reference image, and the second for stereo depth image
    JBUTexObj *jt_d;
    JBUParameters *jp_d;

    void InitializeParameters(int n);
    void CudaRun();
};

#endif // _ACMM_H_
