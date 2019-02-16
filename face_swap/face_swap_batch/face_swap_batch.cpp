// std
#include <iostream>
#include <fstream>
#include <exception>

// Boost
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>
#include <boost/timer/timer.hpp>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// face_swap
#include <face_swap/face_swap_engine.h>
#include <face_swap/utilities.h>
#include <face_swap/render_utilities.h>

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::runtime_error;
using namespace boost::program_options;
using namespace boost::filesystem;

const std::string IMAGE_FILTER =
"(.*\\.(bmp|dib|jpeg|jpg|jpe|jp2|png|pbm|pgm|ppm|sr|ras))";

void nchoose2(const std::vector<string>& in, 
    std::vector<std::pair<string, string>>& out)
{
    size_t n = in.size();
    out.reserve(n*(n - 1));
    int i, j;
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            if(i != j) out.push_back(std::make_pair(in[i], in[j]));
        }
    }
}

void getImagesFromDir(const std::string& dir_path, std::vector<std::string>& img_paths)
{
    boost::regex filter(IMAGE_FILTER);
    boost::smatch what;
    directory_iterator end_itr; // Default ctor yields past-the-end
    for (directory_iterator it(dir_path); it != end_itr; ++it)
    {
        // Skip if not a file
        if (!boost::filesystem::is_regular_file(it->status())) continue;

        // Get extension
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        // Skip if no match
        if (!boost::regex_match(ext, what, filter)) continue;

        img_paths.push_back(it->path().string());
    }
}

void readImagePairsFromFile(const std::string& csv_file,
    std::vector<std::pair<string, string>>& out)
{
    std::ifstream file(csv_file);
    std::pair<string, string> img_pair;
    while (file.good())
    {
        std::getline(file, img_pair.first, ',');
        std::getline(file, img_pair.second, '\n');
        if (img_pair.first.empty() || img_pair.second.empty()) return;
        out.push_back(img_pair);
    }
}

void logError(std::ofstream& log, const std::pair<string, string>& img_pair, 
    const string& msg, bool write_to_file = true)
{
    std::cerr << "Error: " << msg << std::endl;
    if (write_to_file)
    {
        log << img_pair.first << "," << img_pair.second <<
            ",Error: " << msg << std::endl;
    } 
}

int main(int argc, char* argv[])
{
	// Parse command line arguments
    string input_path, seg_path, output_path, landmarks_path;
	string model_3dmm_h5_path, model_3dmm_dat_path;
	string reg_model_path, reg_deploy_path, reg_mean_path;
	string seg_model_path, seg_deploy_path;
    string log_path, cfg_path;
    bool generic, with_expr, with_gpu;
    unsigned int gpu_device_id, verbose;
	try {
		options_description desc("Allowed options");
		desc.add_options()
			("help,h", "display the help message")
            ("verbose,v", value<unsigned int>(&verbose)->default_value(0), "output debug information [0, 4]")
			("input,i", value<string>(&input_path)->required(), "path to input directory or image pairs list")
			("output,o", value<string>(&output_path)->required(), "output directory")
            ("segmentations,s", value<string>(&seg_path)->default_value(""), "segmentations directory")
			("landmarks,l", value<string>(&landmarks_path)->required(), "path to landmarks model file")
			("model_3dmm_h5", value<string>(&model_3dmm_h5_path)->required(), "path to 3DMM file (.h5)")
			("model_3dmm_dat", value<string>(&model_3dmm_dat_path)->required(), "path to 3DMM file (.dat)")
			("reg_model,r", value<string>(&reg_model_path)->required(), "path to 3DMM regression CNN model file (.caffemodel)")
			("reg_deploy,d", value<string>(&reg_deploy_path)->required(), "path to 3DMM regression CNN deploy file (.prototxt)")
			("reg_mean,m", value<string>(&reg_mean_path)->required(), "path to 3DMM regression CNN mean file (.binaryproto)")
			("seg_model", value<string>(&seg_model_path), "path to face segmentation CNN model file (.caffemodel)")
			("seg_deploy", value<string>(&seg_deploy_path), "path to face segmentation CNN deploy file (.prototxt)")
            ("generic,g", value<bool>(&generic)->default_value(false), "use generic model without shape regression")
            ("expressions,e", value<bool>(&with_expr)->default_value(true), "with expressions")
			("gpu", value<bool>(&with_gpu)->default_value(true), "toggle GPU / CPU")
			("gpu_id", value<unsigned int>(&gpu_device_id)->default_value(0), "GPU's device id")
            ("log", value<string>(&log_path)->default_value("face_swap_batch_log.csv"), "log file path")
            ("cfg", value<string>(&cfg_path)->default_value("face_swap_batch.cfg"), "configuration file (.cfg)")
			;
		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).
			positional(positional_options_description().add("input", -1)).run(), vm);

        if (vm.count("help")) {
            cout << "Usage: face_swap_batch [options]" << endl;
            cout << desc << endl;
            exit(0);
        }

        // Read config file
        std::ifstream ifs(vm["cfg"].as<string>());
        store(parse_config_file(ifs, desc), vm);

        notify(vm);

        if (!(is_regular_file(input_path) || is_directory(input_path)))
            throw error("input must be a path to input directory or image pairs list!");
        if(!seg_path.empty() && !is_directory(seg_path))
            throw error("segmentations must be a path to segmentations directory!");
        if ( !is_directory(output_path))
            throw error("output must be a path to a directory!");
		if (!is_regular_file(landmarks_path)) throw error("landmarks must be a path to a file!");
		if (!is_regular_file(model_3dmm_h5_path)) throw error("model_3dmm_h5 must be a path to a file!");
		if (!is_regular_file(model_3dmm_dat_path)) throw error("model_3dmm_dat must be a path to a file!");
		if (!is_regular_file(reg_model_path)) throw error("reg_model must be a path to a file!");
		if (!is_regular_file(reg_deploy_path)) throw error("reg_deploy must be a path to a file!");
		if (!is_regular_file(reg_mean_path)) throw error("reg_mean must be a path to a file!");
		if (!seg_model_path.empty() && !is_regular_file(seg_model_path))
			throw error("seg_model must be a path to a file!");
		if (!seg_deploy_path.empty() && !is_regular_file(seg_deploy_path))
			throw error("seg_deploy must be a path to a file!");
	}
	catch (const error& e) {
        cerr << "Error while parsing command-line arguments: " << e.what() << endl;
        cerr << "Use --help to display a list of options." << endl;
		exit(1);
	}

	try
	{
        // Initialize log file
        std::ofstream log;
        if (verbose > 0)
            log.open(log_path);

        // Parse image pairs
        std::vector<std::pair<string, string>> img_pairs;
        if (is_directory(input_path))
        {
            std::vector<string> img_paths;
            getImagesFromDir(input_path, img_paths);
            nchoose2(img_paths, img_pairs);
        }
        else readImagePairsFromFile(input_path, img_pairs);

		// Initialize face swap
		std::shared_ptr<face_swap::FaceSwapEngine> fs =
			face_swap::FaceSwapEngine::createInstance(
				landmarks_path, model_3dmm_h5_path, model_3dmm_dat_path, reg_model_path,
				reg_deploy_path, reg_mean_path, seg_model_path, seg_deploy_path,
				generic, with_expr, with_gpu, gpu_device_id);

        // Initialize timer
        boost::timer::cpu_timer timer;
        float total_time = 0.0f, fps = 0.0f;
        int frame_counter = 0;

        // For each image pair
        string prev_src_path, prev_tgt_path;
        cv::Mat source_img, target_img, rendered_img;
        for (const auto& img_pair : img_pairs)
        {
            // Check if output image already exists
            path outputName = (path(img_pair.first).stem() += "_") +=
                (path(img_pair.second).stem() += ".jpg");
            string curr_output_path = (path(output_path) /= outputName).string();
            if (is_regular_file(curr_output_path))
            {
                std::cout << "Skipping: " << path(img_pair.first).filename() <<
                    " -> " << path(img_pair.second).filename() << std::endl;
                continue;
            }
            std::cout << "Face swapping: " << path(img_pair.first).filename() <<
                " -> " << path(img_pair.second).filename() << std::endl;

			// Initialize source face data
			face_swap::FaceData src_face_data;
			if (!readFaceData(img_pair.first, src_face_data))
			{
				//src_face_data.enable_seg = true;
				//src_face_data.max_bbox_res = 500;

				// Read source segmentations
				if (seg_model_path.empty() && !seg_path.empty())
				{
					string src_seg_path = (path(seg_path) /=
						(path(img_pair.first).stem() += ".png")).string();
					if (is_regular_file(src_seg_path))
						src_face_data.seg = cv::imread(src_seg_path, cv::IMREAD_GRAYSCALE);
				}
			}

			// Initialize target face data
			face_swap::FaceData tgt_face_data;
			if (!readFaceData(img_pair.second, tgt_face_data))
			{
				//src_face_data.enable_seg = true;
				//src_face_data.max_bbox_res = 500;

				// Read source segmentations
				if (seg_model_path.empty() && !seg_path.empty())
				{
					string src_seg_path = (path(seg_path) /=
						(path(img_pair.second).stem() += ".png")).string();
					if (is_regular_file(src_seg_path))
						tgt_face_data.seg = cv::imread(src_seg_path, cv::IMREAD_GRAYSCALE);
				}
			}

            // Start measuring time
            timer.start();

            // Do face swap
            rendered_img = rendered_img = fs->swap(src_face_data, tgt_face_data);
            if (rendered_img.empty())
            {
                logError(log, img_pair, "Face swap failed!", verbose);
                continue;
            }

            // Stop measuring time
            timer.stop();

            // Write output to file
            std::cout << "Writing " << outputName << " to output directory." << std::endl;
            cv::imwrite(curr_output_path, rendered_img);

            // Print current fps
            total_time += (timer.elapsed().wall*1.0e-9);
            fps = (++frame_counter) / total_time;
            std::cout << "total_time = " << total_time << std::endl;
            std::cout << "fps = " << fps << std::endl;

			// Debug
			if (verbose > 0)
			{
				// Write overlay image
				string debug_overlay_path = (path(output_path) /=
					(path(curr_output_path).stem() += "_overlay.jpg")).string();

				cv::Mat debug_result_img = rendered_img.clone();
				face_swap::renderImageOverlay(debug_result_img, tgt_face_data.scaled_bbox,
					src_face_data.cropped_img, tgt_face_data.cropped_img, cv::Scalar());
				cv::imwrite(debug_overlay_path, debug_result_img);
			}
			if (verbose > 1)
			{
				// Write rendered image
				string debug_render_path = (path(output_path) /=
					(path(curr_output_path).stem() += "_render.jpg")).string();

				cv::Mat src_render = fs->renderFaceData(src_face_data, 3.0f);
				cv::Mat tgt_render = fs->renderFaceData(tgt_face_data, 3.0f);
				cv::Mat debug_render_img;
				int width = std::min(src_render.cols, tgt_render.cols);
				if (src_render.cols > width)
				{
					int height = (int)std::round(src_render.rows * (float(width) / src_render.cols));
					cv::resize(src_render, src_render, cv::Size(width, height));
				}
				else
				{
					int height = (int)std::round(tgt_render.rows * (float(width) / tgt_render.cols));
					cv::resize(tgt_render, tgt_render, cv::Size(width, height));
				}
				cv::vconcat(src_render, tgt_render, debug_render_img);

				cv::imwrite(debug_render_path, debug_render_img);
			}
        }
	}
	catch (std::exception& e)
	{
		cerr << e.what() << endl;
		return 1;
	}

	return 0;
}

