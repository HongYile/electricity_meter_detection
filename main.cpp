#include <opencv2/opencv.hpp>

std::string data_base_path = "/Users/krelar/Documents/Code/electricity_meter_detection/src/elec_watch/";
std::string positive_dir = data_base_path + "positive";
std::string negative_dir = data_base_path + "negative";

void get_hog_descriptor(cv::Mat& image, std::vector<float>& descriptor); // HOG 描述子
void generate_dataset(cv::Mat& train_data, cv::Mat& labels); // 生成数据集
void svm_train(cv::Mat& train_data, cv::Mat& labels); // SVM 训练

int main(int argc, char** argv) {
    // 读取数据 和 生成数据集
    // positive 10张  negative 16张  共26张图片 描述子长度为 3780 = 105 block * 4 cell * 9 bin
    cv::Mat train_data = cv::Mat::zeros(cv::Size(3780, 26), CV_32FC1); // w列数是HOG描述子的个数，h行数是样本数
    cv::Mat labels = cv::Mat::zeros(cv::Size(1, 26), CV_32SC1); // 标签 1 代表正样本 -1 代表负样本
    generate_dataset(train_data, labels); // 生成数据集
    // SVM 训练 和 保存模型
    svm_train(train_data, labels);
    // 加载模型
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(data_base_path + "SVM/svm_model.xml"); // 加载模型
    // 预测
    std::vector<std::string> test_images_paths;
    cv::glob(data_base_path + "test/*.jpg", test_images_paths); // 读取测试图片路径
    for (const auto& test_image_path : test_images_paths) {
        std::string filename = test_image_path.substr(test_image_path.find_last_of("/\\") + 1); // 获取文件名
        std::string output_path = data_base_path + "test_result/" + filename; // 输出路径
        cv::Mat test_img = cv::imread(test_image_path); // 读取测试图片
        if ((test_img.rows > 1024)||(test_img.cols > 1024)) { // 图片尺寸过大 缩放
            cv::resize(
                test_img, test_img,
                cv::Size(0, 0), // 00 代表按比例缩放
                0.2, 0.2 // 横纵方向缩放比例
            );
        }else {
            cv::resize(
                test_img, test_img,
                cv::Size(128, 128), // 1024x1024 代表缩放到指定尺寸
                cv::INTER_AREA // 缩放算法
            );
        }
        cv::Rect winRect;
        winRect.width = 64;
        winRect.height = 128;
        int sum_x = 0, sum_y = 0;
        int count = 0;

        // 开窗检测
        for (int row = 64; row < test_img.rows - 64; row+=4) {
            for (int col = 32; col < test_img.cols - 32; col+=4) {
                winRect.x = col - 32;
                winRect.y = row - 64;
                std::vector<float> fv;
                auto sub_img = test_img(winRect);
                get_hog_descriptor(sub_img, fv); // 获取 HOG 描述子
                cv::Mat one_row = cv::Mat::zeros(cv::Size(fv.size(), 1), CV_32FC1);
                for (int k = 0; k < fv.size(); k++) {
                    one_row.at<float>(0, k) = fv[k];
                }
                float result = svm->predict(one_row); // 预测
                // std::cout << "result: " << result << std::endl;
                if (result > 0) {
                    // cv::rectangle(test_img, winRect, cv::Scalar(0, 0, 255), 1, 8, 0); // 绘制矩形
                    count++;
                    sum_x += winRect.x;
                    sum_y += winRect.y;
                }
            }
        }
        if (count == 0) {
            std::cout << "no electricity meter found in " << filename << std::endl;
            cv::imwrite(output_path, test_img); // 保存输出图片
            continue;
        }
        winRect.x = sum_x / count;
        winRect.y = sum_y / count;
        cv::rectangle(test_img, winRect, cv::Scalar(0, 0, 255), 1, 8, 0); // 绘制矩形
        cv::imwrite(output_path, test_img); // 保存输出图片
        std::cout << "save detect result to " << output_path << std::endl;
    }




    return 0;
}


// HOG 描述子
void get_hog_descriptor(cv::Mat& image, std::vector<float>& descriptor) {
    /*
     * Cell size: 8x8
     * image size: 64x128
     * Number of cells: (64x128)/(8x8) = 8x16 = 128
     * block:cell = 1:4
     * detect_windows_size: 16 cells * 4 cells
     * in rows, one block can move 7 times in 1 stride
     * in cols, one block can move 15 times in 1 stride
     * so, 7*15 = 105 blocks, and 1 block have 36 features(bins) equal 4 cells wtih 9 bins each
     */
    cv::HOGDescriptor hog; // 创建 HOG 描述子对象
    // 获取输入图像的尺寸
    int width = image.cols;
    int height = image.rows;
    float rate = 64.0 / width; // 缩放比例 代表 64x64 的 HOG 块
    cv::Mat img, gray;
    // 缩放图像 使得宽度为 64 像素
    cv::resize(image, img, cv::Size(64, int(height * rate)));
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // 转换为灰度图像
    //
    cv::Mat result = cv::Mat::zeros(cv::Size(64, 128), CV_8UC1);
    result = cv::Scalar(127); // 填充 127 值
    cv::Rect roi;
    roi.x = 0;
    roi.width = 64;
    roi.y = (128 - gray.rows) / 2;
    roi.height = gray.rows;
    gray.copyTo(result(roi)); // 将灰度图像填充到 64x128 的图像中
    hog.compute( // 计算 HOG 描述子
        result, // 输入图像 一定要被2整除（宽高）
        descriptor, // 输出 HOG 描述子
        cv::Size(8, 8), // cell块大小 winStride
        cv::Size(0, 0) // padding 填充 周围填充
    );
    // std::cout << "HOG descriptor len:" << descriptor.size() << std::endl;

}
// 生成数据集
void generate_dataset(cv::Mat& train_data, cv::Mat& labels) {
    // 读取正样本
    std::vector<cv::String> images;
    cv::glob(positive_dir, images);
    int pos_num = images.size();
    for (int i = 0; i < pos_num; i++) {
        cv::Mat img = cv::imread(images[i].c_str());
        std::vector<float> fv;
        get_hog_descriptor(img, fv); // 获取 HOG 描述子
        for (int j = 0; j < fv.size(); j++) {
            train_data.at<float>(i, j) = fv[j];
        }
        labels.at<int>(i, 0) = 1; // 正样本标签为 1
    }
    // 读取负样本
    images.clear();
    cv::glob(negative_dir, images);
    int neg_num = images.size();
    for (int i = 0; i < neg_num; i++) {
        cv::Mat img = cv::imread(images[i].c_str());
        std::vector<float> fv;
        get_hog_descriptor(img, fv); // 获取 HOG 描述子
        for (int j = 0; j < fv.size(); j++) {
            train_data.at<float>(i + pos_num, j) = fv[j];
        }
        labels.at<int>(i + pos_num, 0) = -1; // 负样本标签为 -1
    }
}
// SVM 训练
void svm_train(cv::Mat& train_data, cv::Mat& labels) {
    std::cout << "Start SVM training..." << std::endl;
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create(); // 创建 SVM 对象
    svm->setC(2.67); // 设置参数 C  这个值越大，对误分类的惩罚越大，越容易过拟合 默认2.67
    svm->setType(cv::ml::SVM::C_SVC); // 设置 SVM 类型为 C_SVC 是指支持向量机
    svm->setKernel(cv::ml::SVM::LINEAR); // 设置核函数为线性核 速度快，精度低 其他核如 RBF 高精度，但速度慢  默认径向基函数 RBF 核  gamma=0.5
    svm->setGamma(5.383); // 设置 gamma 值  gamma 值越大，对样本的影响越小，越容易过拟合  默认0.5
    svm->train(// 训练 SVM 模型
        train_data, // 训练数据
        cv::ml::ROW_SAMPLE, // 按行组织数据
        labels // 标签
    );
    std::cout << "SVM training finished." << std::endl;
    svm->save(data_base_path + "SVM/svm_model.xml"); // 保存模型
}