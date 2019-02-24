#include "face_swap/face_swap_engine_impl.h"
#include "face_swap/utilities.h"
#include "face_swap/landmarks_utilities.h"

// std
#include <limits>
#include <iostream> // debug
#include <chrono>

// OpenCV
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  // Debug
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/miniflann.hpp>

using sclock = std::chrono::system_clock;
using ms = std::chrono::milliseconds;

namespace face_swap
{
	std::shared_ptr<FaceSwapEngine> FaceSwapEngine::createInstance(
		const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
		const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
		const std::string& reg_deploy_path, const std::string& reg_mean_path,
		const std::string& seg_model_path, const std::string& seg_deploy_path,
		bool generic, bool with_expr, bool with_gpu, int gpu_device_id)
	{
		return std::make_shared<FaceSwapEngineImpl>(
			landmarks_path, model_3dmm_h5_path,
			model_3dmm_dat_path, reg_model_path,
			reg_deploy_path, reg_mean_path,
			seg_model_path, seg_deploy_path,
			generic, with_expr, with_gpu, gpu_device_id);
	}

	FaceSwapEngineImpl::FaceSwapEngineImpl(
		const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
		const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
		const std::string& reg_deploy_path, const std::string& reg_mean_path,
		const std::string& seg_model_path, const std::string& seg_deploy_path,
		bool generic, bool with_expr, bool with_gpu, int gpu_device_id) :
		m_with_gpu(with_gpu),
		m_gpu_device_id(gpu_device_id)
	{
		// Initialize Sequence Face Landmarks
		//m_sfl = sfl::SequenceFaceLandmarks::create(landmarks_path);

		// Initialize detection and landmarks
		m_lms = FaceDetectionLandmarks::create(landmarks_path);

		// Initialize CNN 3DMM with exression
		m_cnn_3dmm_expr = std::make_unique<CNN3DMMExpr>(
			reg_deploy_path, reg_model_path, reg_mean_path, model_3dmm_dat_path,
			generic, with_expr, with_gpu, gpu_device_id);

		// Initialize segmentation model
		if (!(seg_model_path.empty() || seg_deploy_path.empty()))
			m_face_seg = std::make_unique<FaceSeg>(seg_deploy_path,
				seg_model_path, with_gpu, gpu_device_id, true, true);

		// Load Basel 3DMM
		m_basel_3dmm = std::make_unique<Basel3DMM>();
		*m_basel_3dmm = Basel3DMM::load(model_3dmm_h5_path);
	}

	cv::Mat FaceSwapEngineImpl::swap(FaceData& src_data, FaceData& tgt_data)
	{
		// Process images
		process(src_data);
		process(tgt_data);

		// Check if horizontal flip is required
		float src_angle = getFaceApproxHorAngle(src_data.cropped_landmarks);
		float tgt_angle = getFaceApproxHorAngle(tgt_data.cropped_landmarks);
		cv::Mat cropped_src, cropped_src_seg;
		std::vector<cv::Point> cropped_src_landmarks = src_data.cropped_landmarks;
		cv::Mat src_shape_coefficients, src_tex_coefficients, src_expr_coefficients;
		cv::Mat src_vecR, src_vecT;
		cv::Mat src_K = src_data.K;
		if ((src_angle * tgt_angle) < 0 && std::abs(src_angle - tgt_angle) > (CV_PI / 18.0f) &&
            std::abs(src_angle) > (CV_PI / 36.0f))
		{
			// Horizontal flip the source image
			cv::flip(src_data.cropped_img, cropped_src, 1);
			if (!src_data.cropped_seg.empty())
				cv::flip(src_data.cropped_seg, cropped_src_seg, 1);

			// Horizontal flip the source landmarks
			horFlipLandmarks(cropped_src_landmarks, cropped_src.cols);

			// Recalculate source coefficients
			if (src_data.shape_coefficients_flipped.empty() || src_data.expr_coefficients_flipped.empty())
			{
				m_cnn_3dmm_expr->process(cropped_src, cropped_src_landmarks,
					src_data.shape_coefficients_flipped,
					src_data.tex_coefficients_flipped, src_data.expr_coefficients_flipped,
					src_data.vecR_flipped, src_data.vecT_flipped, src_data.K);
			}

			src_shape_coefficients = src_data.shape_coefficients_flipped;
			src_tex_coefficients = src_data.tex_coefficients_flipped;
			src_expr_coefficients = src_data.expr_coefficients_flipped;
			src_vecR = src_data.vecR_flipped;
			src_vecT = src_data.vecT_flipped;
		}
		else
		{
			cropped_src = src_data.cropped_img;
			cropped_src_seg = src_data.cropped_seg;
			src_shape_coefficients = src_data.shape_coefficients;
			src_tex_coefficients = src_data.tex_coefficients;
			src_expr_coefficients = src_data.expr_coefficients;
			src_vecR = src_data.vecR;
			src_vecT = src_data.vecT;
		}

		// Source mesh
		Mesh src_mesh;
		cv::Mat src_tex, src_uv;
		{
			// Create source mesh
			src_mesh = m_basel_3dmm->sample(src_shape_coefficients, src_tex_coefficients,
				src_expr_coefficients);

			// Texture source mesh
			generateTexture(src_mesh, cropped_src, cropped_src_seg, src_vecR, src_vecT, src_K,
				src_tex, src_uv);
		}


		// Create target mesh
		Mesh tgt_mesh = m_basel_3dmm->sample(tgt_data.shape_coefficients,
			tgt_data.tex_coefficients, tgt_data.expr_coefficients);
		tgt_mesh.tex = src_tex;
		tgt_mesh.uv = src_uv;

		////////////////////////////////////////
		// Actual swap
		////////////////////////////////////////

		// Render
		cv::Mat rendered_img = tgt_data.cropped_img.clone();
		cv::Mat depthbuf;
		renderMesh(rendered_img, tgt_mesh, tgt_data.vecR, tgt_data.vecT, tgt_data.K, depthbuf);

		// Copy back to original target image
		cv::Mat tgt_rendered_img = tgt_data.scaled_img.clone();
		rendered_img.copyTo(tgt_rendered_img(tgt_data.scaled_bbox));
		cv::Mat tgt_depthbuf(tgt_data.scaled_img.size(), CV_32F, std::numeric_limits<float>::max());
		depthbuf.copyTo(tgt_depthbuf(tgt_data.scaled_bbox));

		// Create binary mask from the rendered depth buffer
		cv::Mat mask(tgt_depthbuf.size(), CV_8U);
		unsigned char* mask_data = mask.data;
		float* tgt_depthbuf_data = (float*)tgt_depthbuf.data;
		for (int i = 0; i < tgt_depthbuf.total(); ++i)
		{
			if ((*tgt_depthbuf_data++ - 1e-6f) < std::numeric_limits<float>::max())
				*mask_data++ = 255;
			else *mask_data++ = 0;
		}

		// Combine the segmentation with the mask
		if (!tgt_data.scaled_seg.empty())
			cv::bitwise_and(mask, tgt_data.scaled_seg, mask);

		// Blend images
		return blend(tgt_rendered_img, tgt_data.scaled_img, mask);
	}

	bool FaceSwapEngineImpl::process(FaceData& face_data, bool process_flipped)
	{
		auto before = sclock::now();
		// Preprocess input image
		if (face_data.scaled_landmarks.empty())
		{
			if (!preprocessImages(face_data))
				return false;
		}
		auto duration = std::chrono::duration_cast<ms>(sclock::now() - before);
		std::cout << "During face data processing, prepocessImages took " << duration.count()/1000.0 << "ms" << std::endl;

		before = sclock::now();
		// If segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		bool compute_seg = face_data.scaled_seg.empty() && face_data.enable_seg && m_face_seg != nullptr;
		if (compute_seg)
		{
			face_data.cropped_seg = m_face_seg->process(face_data.cropped_img);
			face_data.scaled_seg = cv::Mat::zeros(face_data.scaled_img.size(), CV_8U);
			face_data.cropped_seg.copyTo(face_data.scaled_seg(face_data.scaled_bbox));
		}
		cv::imwrite("cropped_seg.jpg", face_data.cropped_seg);
		cv::imwrite("cropped_img.jpg", face_data.cropped_img);
		duration = std::chrono::duration_cast<ms>(sclock::now() - before);
		std::cout << "During face data processing, face segmentation took " << duration.count()/1000.0 << "ms" << std::endl;

		before = sclock::now();
		// Calculate coefficients and pose
		if (face_data.shape_coefficients.empty() || face_data.expr_coefficients.empty())
		{
			m_cnn_3dmm_expr->process(face_data.cropped_img, face_data.cropped_landmarks,
				face_data.shape_coefficients, face_data.tex_coefficients,
				face_data.expr_coefficients, face_data.vecR, face_data.vecT, face_data.K);
		}
		duration = std::chrono::duration_cast<ms>(sclock::now() - before);
		std::cout << "During face data processing, cnn3dmm took " << duration.count()/1000.0 << "ms" << std::endl;

		// Calculate flipped coefficients and pose
		if (process_flipped && (face_data.shape_coefficients_flipped.empty() ||
			face_data.expr_coefficients_flipped.empty()))
		{
			// Horizontal flip the cropped image
			cv::Mat cropped_img_flipped;
			cv::flip(face_data.cropped_img, cropped_img_flipped, 1);

			// Horizontal flip the cropped landmarks
			std::vector<cv::Point> cropped_landmarks_flipped = face_data.cropped_landmarks;
			horFlipLandmarks(cropped_landmarks_flipped, cropped_img_flipped.cols);

			// Recalculate source coefficients
			m_cnn_3dmm_expr->process(cropped_img_flipped, cropped_landmarks_flipped,
				face_data.shape_coefficients_flipped,
				face_data.tex_coefficients_flipped, face_data.expr_coefficients_flipped,
				face_data.vecR_flipped, face_data.vecT_flipped, face_data.K);
		}
			
		return true;
	}

	cv::Mat FaceSwapEngineImpl::renderFaceData(const FaceData& face_data, float scale)
	{
		cv::Mat out = face_data.scaled_img.clone();
		if(scale != 1.0f)
			cv::resize(out, out, cv::Size(), scale, scale, cv::INTER_CUBIC);

		// Check if a face was detected
		if (face_data.scaled_landmarks.empty())
		{
			cv::Mat temp = out.clone();
			cv::hconcat(out, temp, out);
			cv::hconcat(out, temp, out);
			return out;
		}

		// Render landmarks and bounding box
		std::vector<cv::Point> landmarks = face_data.scaled_landmarks;
		cv::Rect bbox = face_data.scaled_bbox;
		if (scale != 1.0f)
		{
			// Scale landmarks
			for (auto&& p : landmarks)
			{
				p.x = (int)std::round(p.x * scale);
				p.y = (int)std::round(p.y * scale);
			}

			// Scale bounding box
			bbox.x = (int)std::round(face_data.scaled_bbox.x * scale);
			bbox.y = (int)std::round(face_data.scaled_bbox.y * scale);
			bbox.width = (int)std::round(face_data.scaled_bbox.width * scale);
			bbox.height = (int)std::round(face_data.scaled_bbox.height * scale);
		}
		cv::Mat landmarks_render = face_data.scaled_img.clone();
		if (scale != 1.0f)
			cv::resize(landmarks_render, landmarks_render, cv::Size(), scale, scale, cv::INTER_CUBIC);
		int thickness = int(bbox.width / 200.0f);
		render(landmarks_render, landmarks, false, cv::Scalar(0, 255, 0), thickness);
		render(landmarks_render, bbox, cv::Scalar(0, 0, 255), thickness);
		out = landmarks_render;

		// Render mesh wireframe
		cv::Mat wireframe_render = face_data.scaled_img.clone();
		if (scale != 1.0f)
			cv::resize(wireframe_render, wireframe_render, cv::Size(), scale, scale, cv::INTER_CUBIC);
		cv::Mat wireframe_render_cropped = face_data.cropped_img.clone();
		cv::Mat P = createPerspectiveProj3x4(face_data.vecR, face_data.vecT, face_data.K);
		Mesh mesh = m_basel_3dmm->sample(face_data.shape_coefficients,
			face_data.tex_coefficients, face_data.expr_coefficients);
		renderWireframe(wireframe_render_cropped, mesh, P, scale);
		wireframe_render_cropped.copyTo(wireframe_render(bbox));
		cv::hconcat(out, wireframe_render, out);

		// Render segmentation
		cv::Mat seg_render = face_data.scaled_img.clone();
		if (!face_data.scaled_seg.empty())
			renderSegmentation(seg_render, face_data.scaled_seg);
		if (scale != 1.0f)
			cv::resize(seg_render, seg_render, cv::Size(), scale, scale, cv::INTER_CUBIC);
		cv::hconcat(out, seg_render, out);

		return out;
	}

	bool FaceSwapEngineImpl::preprocessImages(FaceData& face_data)
	{
		// Calculate landmarks
		//m_sfl->clear();
		//const sfl::Frame& lmsFrame = m_sfl->addFrame(face_data.img);
		//if (lmsFrame.faces.empty()) return false;
		////std::cout << "faces found = " << lmsFrame.faces.size() << std::endl;    // Debug
		//const sfl::Face* face = lmsFrame.getFace(sfl::getMainFaceID(m_sfl->getSequence()));

		std::vector<Face> faces;
		m_lms->process(face_data.img, faces);
		if (faces.empty()) return false;
		Face& main_face = faces[getMainFaceID(faces, face_data.img.size())];
		face_data.scaled_landmarks = main_face.landmarks;

		// Calculate crop bounding box
		face_data.bbox = getFaceBBoxFromLandmarks(face_data.scaled_landmarks, face_data.img.size(), true);
		face_data.bbox.width = face_data.bbox.width / 4 * 4;    // Make sure cropped image is dividable by 4
		face_data.bbox.height = face_data.bbox.height / 4 * 4;
		
		// Inforce maximum bounding box resolution
		if (face_data.max_bbox_res > 0 && face_data.max_bbox_res < face_data.bbox.width)
		{
			float scale = (float)face_data.max_bbox_res / (float)face_data.bbox.width;

			// Scale landmarks
			for (cv::Point& p : face_data.scaled_landmarks)
			{
				p.x = (int)std::round((float)p.x * scale);
				p.y = (int)std::round((float)p.y * scale);
			}

			// Scale bounding box
			face_data.scaled_bbox.x = (int)std::round((float)face_data.bbox.x * scale);
			face_data.scaled_bbox.y = (int)std::round((float)face_data.bbox.y * scale);
			face_data.scaled_bbox.width = (int)std::round((float)face_data.bbox.width * scale);
			face_data.scaled_bbox.height = (int)std::round((float)face_data.bbox.height * scale);
			face_data.scaled_bbox.width = face_data.scaled_bbox.width / 4 * 4;    // Make sure cropped image is dividable by 4
			face_data.scaled_bbox.height = face_data.scaled_bbox.height / 4 * 4;

			// Scale images
			cv::resize(face_data.img, face_data.scaled_img, cv::Size(), scale, scale, cv::INTER_CUBIC);
			if (!face_data.seg.empty())
				cv::resize(face_data.seg, face_data.scaled_seg, cv::Size(), scale, scale, cv::INTER_CUBIC);
		}
		else
		{
			face_data.scaled_bbox = face_data.bbox;
			face_data.scaled_img = face_data.img;
			face_data.scaled_seg = face_data.seg;
		}

		// Crop landmarks
		face_data.cropped_landmarks = face_data.scaled_landmarks;
		for (cv::Point& p : face_data.cropped_landmarks)
		{
			p.x -= face_data.scaled_bbox.x;
			p.y -= face_data.scaled_bbox.y;
		}

		// Crop images
		face_data.cropped_img = face_data.scaled_img(face_data.scaled_bbox);
		if (!face_data.scaled_seg.empty()) 
			face_data.cropped_seg = face_data.scaled_seg(face_data.scaled_bbox);

		return true;
	}


	bool FaceSwapEngineImpl::estimate(FaceData& face_data)
	{
		// preprocess
		preprocessImages(face_data);

		// face segmentation
		segment(face_data);

		// 2d landmarks
		landmarks(face_data);

	//	// 3d shape coefficients and pose
	//	shape(face_data);


		return true;
	}

	bool FaceSwapEngineImpl::compare(FaceData& src_data, FaceData& tgt_data)
	{
		// compare 2d segmentation map and 3d shape and pose
		return true;
	}

	cv::Mat FaceSwapEngineImpl::align(FaceData& src_data,  FaceData& tgt_data)
	{
		// convex hull
		std::vector<cv::Point> hull_src;
		std::vector<cv::Point> hull_tgt;
		std::vector<int> hull_idx;
		convexHull(tgt_data.cropped_landmarks, hull_idx, false, false);
		for(int i = 0; i < hull_idx.size(); i++)
		{
			cv::Point src_lm = src_data.cropped_landmarks[hull_idx[i]];
			cv::Point tgt_lm = tgt_data.cropped_landmarks[hull_idx[i]];
			hull_src.push_back(cv::Point(src_lm.x, src_lm.y));
			hull_tgt.push_back(cv::Point(tgt_lm.x, tgt_lm.y));
			//std::cout << hull_idx[i] <<  " ";
		}
		//std::cout << std::endl;
		//std::cout << hull_src << std::endl;

		//cv::Mat t = cv::estimateAffinePartial2D(hull_src, hull_tgt);
		cv::Mat t;
		computeRigid(hull_src, hull_tgt, t);
	//	t.at<double>(0, 1) = - t.at<double>(0, 1);
	//	t.at<double>(1, 0) = - t.at<double>(1, 0);
	//	std::cout << t << std::endl;
	//	cv::Mat warpped;
	//	cv::warpAffine(src_data.cropped_img, warpped, t, cv::Size(tgt_data.cropped_img.cols, tgt_data.cropped_img.rows));
	//	writeImage("warpped.jpg", warpped);

		writeImage("src_seg.png",src_data.cropped_seg);
		writeImage("src_face.png",src_data.cropped_img);
		writeImage("tgt_seg.png",tgt_data.cropped_seg);
		writeImage("tgt_face.png",tgt_data.cropped_img);

		return t;
	}

	cv::Mat FaceSwapEngineImpl::fine_tune(cv::Mat& src_data, cv::Mat& tgt_data)
	{
		// extract mask boundary
		auto mask_boundary = [](cv::Mat mask)
		{
			std::vector<std::vector<cv::Point>> contours;
			std::vector<cv::Point> totalContours;
			cv::findContours(mask, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);		
			//for (auto contour : contours)
			//	totalContours.insert(totalContours.end(), contour.begin(), contour.end());

			//return totalContours;
			return contours[0];
		};

		auto source = mask_boundary(src_data);
		auto target = mask_boundary(tgt_data);

		//for (int i = 0; i < src_data.cropped_landmarks.size(); ++i)
		//	std::cout << "2d lm:" << src_data.cropped_landmarks[i].x << "," << src_data.cropped_landmarks[i].y << std::endl;

		cv::Mat t;
		//icp(src_data.cropped_landmarks, tgt_data.cropped_landmarks, t);
		icp(source, target, t);
		
		return t;
	}

	cv::Mat FaceSwapEngineImpl::transfer(FaceData& src_data, FaceData& tgt_data)
	{
		// align source face mask into target face mask using 2d landmarks
		cv::Mat aligned_mat = align(src_data, tgt_data);

		// optimization with landmarks and edge map of segmentation mask
		cv::Mat warpped_src_img, warpped_src_seg;
		cv::warpAffine(src_data.cropped_img, warpped_src_img, aligned_mat.rowRange(0,2), cv::Size(tgt_data.cropped_img.cols, tgt_data.cropped_img.rows));
		cv::warpAffine(src_data.cropped_seg, warpped_src_seg, aligned_mat.rowRange(0,2), cv::Size(tgt_data.cropped_img.cols, tgt_data.cropped_img.rows));
		writeImage("warpped_src_img.jpg", warpped_src_img);
		writeImage("warpped_src_seg.jpg", warpped_src_seg);
		cv::Mat tuned_mat = fine_tune(warpped_src_seg, tgt_data.cropped_seg);

		// blending
		cv::Mat t = tuned_mat * aligned_mat;
		auto mask_center = [](cv::Mat mask)
		{
			int h = mask.rows;
			int w = mask.cols;
			cv::Mat gray = cv::Mat(mask.size(), CV_8UC1);
			if(mask.channels() == 3)
				cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
			else
				gray = mask;

    		int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;
			for(int i=0;i<h;i++)
			{
				for(int j=0;j<w;j++)
				{
					if(gray.at<uchar>(i,j) == 255)
					{
						minx = std::min(minx,i);
						maxx = std::max(maxx,i);
						miny = std::min(miny,j);
						maxy = std::max(maxy,j);
					}
				}
			}

			return cv::Point((miny + maxy)/2, (minx + maxx)/2);
		};

		cv::Mat warpped_img, warpped_seg;
		cv::warpAffine(src_data.cropped_img, warpped_img, t.rowRange(0,2), cv::Size(tgt_data.cropped_img.cols, tgt_data.cropped_img.rows));
		cv::warpAffine(src_data.cropped_seg, warpped_seg, t.rowRange(0,2), cv::Size(tgt_data.cropped_img.cols, tgt_data.cropped_img.rows));
		writeImage("warpped_img.jpg", warpped_img);
		writeImage("warpped_seg.jpg", warpped_seg);

		cv::Mat blended;
		cv::Point p = mask_center(tgt_data.cropped_seg);
		std::cout << "center:" << p << std::endl;
		cv::seamlessClone(warpped_img, tgt_data.cropped_img, warpped_seg, p, blended, cv::NORMAL_CLONE);
		writeImage("cloned.jpg", blended);

		return cv::Mat();
	}


	bool FaceSwapEngineImpl::segment(FaceData& face_data)
	{
		face_data.cropped_seg = m_face_seg->process(face_data.cropped_img);
		face_data.scaled_seg = cv::Mat::zeros(face_data.scaled_img.size(), CV_8U);
		face_data.cropped_seg.copyTo(face_data.scaled_seg(face_data.scaled_bbox));
		return true;
	}

	bool FaceSwapEngineImpl::landmarks(FaceData& face_data)
	{
		std::vector<Face> faces;
		m_lms->process(face_data.img, faces);
		return true;
	}

	bool FaceSwapEngineImpl::shape(FaceData& face_data)
	{
		m_cnn_3dmm_expr->process(face_data.cropped_img, face_data.cropped_landmarks,
			face_data.shape_coefficients, face_data.tex_coefficients,
			face_data.expr_coefficients, face_data.vecR, face_data.vecT, face_data.K);
		return true;
	}

	// https://stackoverflow.com/questions/21206870/opencv-rigid-transformation-between-two-3d-point-clouds
	// https://docs.opencv.org/3.1.0/d4/d61/tutorial_warp_affine.html
	bool FaceSwapEngineImpl::computeRigid(const std::vector<cv::Point> &srcPoints, const std::vector<cv::Point> &dstPoints, cv::Mat &transf, bool xyExchange)
	{
		// sanity check
		if ((srcPoints.size() < 2) || (srcPoints.size() != dstPoints.size()))
			return false;
		
		auto convert2Mat = [](const std::vector<cv::Point>& points, bool xyExchange = false)
		{
			cv::Mat_<cv::Vec2d> m(cv::Size(1, points.size()));
			for (int i = 0; i < points.size(); ++ i)
			{
				if (xyExchange)
					m(i) = cv::Vec2d(points[i].y, points[i].x); // here exchange x and y, because later in AffineWarp a positive angle is counter-clockwise
				else 
					m(i) = cv::Vec2d(points[i].x, points[i].y); 
			}
			return m;
		};

		cv::Mat_<cv::Vec2d> source = convert2Mat(srcPoints, xyExchange);
		cv::Mat_<cv::Vec2d> target = convert2Mat(dstPoints, xyExchange);

		auto calMean = [](const cv::Mat_<cv::Vec2d>& points)
		{
			cv::Mat_<cv::Vec2d> result;
	    	cv::reduce(points, result, 0, CV_REDUCE_AVG);

	//		std::cout << points << std::endl;
	//		std::cout << result<< std::endl;
    		return result(0, 0);
		};

		auto centeredPoints = [](const cv::Mat_<cv::Vec2d>& points, cv::Vec2d c)
		{
			return points - c;
		};

		/* Calculate centroids. */
		cv::Vec2d c1 = calMean(source);
		cv::Vec2d c2 = calMean(target);

		cv::Mat_<double> T1 = cv::Mat_<double>::eye(3, 3);
		T1(0, 2) = -c1[0];
		T1(1, 2) = -c1[1];

		cv::Mat_<double> T2 = cv::Mat_<double>::eye(3, 3);
		T2(0, 2) = c2[0];
		T2(1, 2) = c2[1];

		/* Calculate covariance matrix for input points. Also calculate RMS deviation from centroid
		* which is used for scale calculation.
		*/
		cv::Mat_<cv::Vec2d> srcM = centeredPoints(source, c1);
		cv::Mat_<cv::Vec2d> dstM = centeredPoints(target, c2);
		cv::Mat_<double> C(2, 2, 0.0);
		double p1Rms = 0, p2Rms = 0;
		for (int ptIdx = 0; ptIdx < srcM.rows; ptIdx++) {
			cv::Vec2d p1 = srcM(ptIdx);
			cv::Vec2d p2 = dstM(ptIdx);
			p1Rms += p1.dot(p1);
			p2Rms += p2.dot(p2);
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					C(i, j) += p1[i] * p2[j];
				}
			}
		}

		cv::Mat_<double> u, s, vh;
		cv::SVD::compute(C, s, u, vh);

		cv::Mat_<double> R = u * vh;

		if (cv::determinant(R) < 0) {
			R -= u.col(1) * (vh.row(1) * 2.0);
		}

		double scale = sqrt(p2Rms / p1Rms);
    	R *= scale;

    	cv::Mat_<double> M = cv::Mat_<double>::eye(3, 3);
    	R.copyTo(M.colRange(0, 2).rowRange(0, 2));

		cv::Mat_<double> result = T2 * M * T1;

//		transf = result.rowRange(0, 2);
		transf = result;

//		std::cout << source << std::endl;
//		std::cout << srcM << std::endl;
//		std::cout << C << std::endl;
//		std::cout << R << std::endl;
//		std::cout << T2 << std::endl;
//		std::cout << M << std::endl;
//		std::cout << T1 << std::endl;
//		std::cout << transf << std::endl;

		return true;
	}

	bool FaceSwapEngineImpl::icp(const std::vector<cv::Point> &srcPoints, const std::vector<cv::Point> &dstPoints, cv::Mat &transf)
	{
		// sanity check
		if ((srcPoints.size() < 2) || (dstPoints.size() < 2))
			return false;
		
		auto convert2Mat = [](const std::vector<cv::Point>& points, bool xyExchange = true)
		{
			cv::Mat_<float> m(cv::Size(2, points.size()));
			for (int i = 0; i < points.size(); ++ i)
			{
				if (xyExchange)
				{
					// here exchange x and y, because later in AffineWarp a positive angle is counter-clockwise
					m.at<float>(i, 0) = (float)points[i].y; 
					m.at<float>(i, 1) = (float)points[i].x; 
				}
				else 
				{
					m.at<float>(i, 0) = (float)points[i].x; 
					m.at<float>(i, 1) = (float)points[i].y; 
				}
			}

			return m;
		};

		cv::Mat_<float> source = convert2Mat(srcPoints);
		cv::Mat_<float> target = convert2Mat(dstPoints);

		auto knn = [](cv::Mat_<float> source, cv::flann::Index& kdtree)
		{
			// find knn
			cv::Mat indices, dists;
			kdtree.knnSearch(source, indices, dists, 1);

		//	std::cout << source.at<float>(0, 0) << "," << source.at<float>(0,1) << std::endl<< std::endl;
		//	std::cout << "knn indices:" << indices.cols << "," << indices.rows << std::endl;

			return indices;
		};

		auto convert = [](cv::Mat_<float> source, cv::Mat_<float> target, cv::Mat indices)
		{
			std::vector<cv::Point> srcPoints, dstPoints;
			for (int i = 0; i < indices.rows; ++ i)
			{
				srcPoints.push_back(cv::Point((int)source.at<float>(i, 0), (int)source.at<float>(i, 1)));
				dstPoints.push_back(cv::Point((int)target.at<float>(indices.at<int>(i,0), 0), (int)target.at<float>(indices.at<int>(i,0), 1)));
			}

			return std::make_tuple(srcPoints, dstPoints);
		};

		int iterNum = 10;
		int loop = 0;	
		cv::Mat_<double> total = cv::Mat_<double>::eye(3, 3);

		cv::flann::Index kdtree(target, cv::flann::KDTreeIndexParams(1));
		do
		{
			// build correspondences
			auto indices = knn(source, kdtree);
			auto tuple = convert(source, target, indices);

			// compute rigid transform
			cv::Mat_<double> t;
			auto srcP = std::get<0>(tuple);
			auto dstP = std::get<1>(tuple);
			computeRigid(srcP, dstP, t, false);

			std::cout << "after rigid t:" << std::endl;
			// transform source 

			std::cout << "before:" << srcP[0].x << "," << srcP[0].y << std::endl;
			std::cout << t << std::endl;
			cv::transform(srcP, srcP, t.rowRange(0,2));
			std::cout << "after:" << srcP[0].x << "," << srcP[0].y << std::endl;
		//	std::cout << "after transform" << std::endl;
			total = t * total;
		//	std::cout << "after accu" << std::endl;
			source = convert2Mat(srcP, false);

			loop ++;	
		}while(loop < iterNum);
	
		transf = total;

		std::cout << "icp transform" << std::endl;
		std::cout << transf << std::endl;

		return true;
	}

}   // namespace face_swap