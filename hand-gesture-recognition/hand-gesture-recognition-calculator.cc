#include <cmath>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

#include "protobuf/sensors.pb.h"
#include "protobuf/discovery.pb.h"

#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <thread>

//#define PORT 5000
//#define IP "192.168.22.172"

#define PORT 8915
#define IP "192.168.22.65"

namespace mediapipe
{

	namespace
	{
		constexpr char normRectTag[] = "NORM_RECT";
		constexpr char normalizedLandmarkListTag[] = "NORM_LANDMARKS";
		constexpr char landmarkListTag[] = "LANDMARKS";
		constexpr char recognizedHandMouvementScrollingTag[] = "RECOGNIZED_HAND_GESTURE";
	}

	class HandGestureRecognitionCalculator : public CalculatorBase
	{
	public:
		HandGestureRecognitionCalculator()
		{
			if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
			{
				LOG(ERROR) << "Socket creation error";
			}

			serv_addr.sin_family = AF_INET;
			serv_addr.sin_port = htons(PORT);

			if (inet_pton(AF_INET, IP, &serv_addr.sin_addr) <= 0)
			{
				LOG(ERROR) << "Invalid address";
			}

			reconnect(true);
		}

		static ::mediapipe::Status GetContract(CalculatorContract *cc);
		::mediapipe::Status Open(CalculatorContext *cc) override;

		::mediapipe::Status Process(CalculatorContext *cc) override;

	private:
		float previous_x_center;
		float previous_y_center;
		float previous_angle;
		float previous_rectangle_width;
		float previous_rectangle_height;
		float last_x;
		float last_y;

		int last_gesture = 0;

		const float thumb_index_distance_threshold = 0.03;
		const float index_middle_distance_threshold = 0.06;
		const float movementDistanceFactor = 0.15; // movement threshold.
		std::chrono::duration<double> max_frame_time_difference = std::chrono::milliseconds(100);
		std::chrono::duration<double> min_gesture_time_difference = std::chrono::milliseconds(1000);

		std::chrono::time_point<std::chrono::steady_clock> previous_frame_time;
		std::chrono::time_point<std::chrono::steady_clock> previous_gesture_time;

		int sock = 0, client_fd;
		struct sockaddr_in serv_addr;

		using Time = std::chrono::steady_clock;
		using float_sec = std::chrono::duration<float>;
		using float_time_point = std::chrono::time_point<Time, float_sec>;

		bool reconnecting = false;
		float_time_point reconnect_start;

		void sendAllSensorDiscovery()
		{
			std::unordered_map<std::string, naki3d::common::protocol::DataType> typeMap =
				{
					{"center_position", naki3d::common::protocol::DataType::Vector3},

					{"gestures/swipe_left", naki3d::common::protocol::DataType::Void},
					{"gestures/swipe_right", naki3d::common::protocol::DataType::Void},
					{"gestures/swipe_up", naki3d::common::protocol::DataType::Void},
					{"gestures/swipe_down", naki3d::common::protocol::DataType::Void},
					{"gestures/close_hand", naki3d::common::protocol::DataType::Void},
					{"gestures/open_hand", naki3d::common::protocol::DataType::Void},
					{"gestures/pinch", naki3d::common::protocol::DataType::Void},

					{"fingers/thumb/closed", naki3d::common::protocol::DataType::Bool},
					{"fingers/thumb/position", naki3d::common::protocol::DataType::Vector3},
					{"fingers/index/closed", naki3d::common::protocol::DataType::Bool},
					{"fingers/index/position", naki3d::common::protocol::DataType::Vector3},
					{"fingers/middle/closed", naki3d::common::protocol::DataType::Bool},
					{"fingers/middle/position", naki3d::common::protocol::DataType::Vector3},
					{"fingers/ring/closed", naki3d::common::protocol::DataType::Bool},
					{"fingers/ring/position", naki3d::common::protocol::DataType::Vector3},
					{"fingers/pinky/closed", naki3d::common::protocol::DataType::Bool},
					{"fingers/pinky/position", naki3d::common::protocol::DataType::Vector3},
				};

			for (auto &type : typeMap)
			{
				sendSensorDiscovery("mediapipe/handtracking/hand/left/" + type.first, type.second);
				sendSensorDiscovery("mediapipe/handtracking/hand/right/" + type.first, type.second);
			}
		}

		void sendSensorDiscovery(const std::string &path, naki3d::common::protocol::DataType type)
		{
			naki3d::common::protocol::SensorDescriptor *descriptor = new naki3d::common::protocol::SensorDescriptor();
			descriptor->set_path(path);
			descriptor->set_model("mediapipe v0.7");
			descriptor->set_data_type(type);

			naki3d::common::protocol::SensorMessage *message = new naki3d::common::protocol::SensorMessage();
			message->set_allocated_sensor_descriptor(descriptor);

			size_t size = message->ByteSizeLong();
			char *buffer = new char[size + 2];
			buffer[0] = static_cast<uint8_t>(size);
			buffer[0] |= static_cast<uint8_t>(0x80);
			buffer[1] = static_cast<uint8_t>(size >> 7);
			message->SerializeToArray(buffer + 2, size);
			if (send(sock, buffer, size + 2, MSG_NOSIGNAL) < 0)
			{
				reconnect();
			}
			delete[] buffer;
		}

		void reconnect(bool initial_connection = false)
		{
			if (!reconnecting)
			{
				if (!initial_connection)
				{
					LOG(ERROR) << "Broken connection";
				}
				reconnecting = true;
				reconnect_start = Time::now();
			}
			else if (Time::now() - reconnect_start > float_sec(1))
			{
				reconnect_start = Time::now();
			}
			else
			{
				return;
			}

			close(sock);
			sock = socket(AF_INET, SOCK_STREAM, 0);
			LOG(INFO) << "Conecting...";
			if ((client_fd = connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr))) == 0)
			{
				LOG(INFO) << "Socket connected";
				LOG(INFO) << "Sending discovery message";
				sendAllSensorDiscovery();
				reconnecting = false;
			}
		}

		void sendMessage(naki3d::common::protocol::SensorDataMessage *data)
		{
			naki3d::common::protocol::SensorMessage *message = new naki3d::common::protocol::SensorMessage();
			message->set_allocated_data(data);

			size_t size = message->ByteSizeLong();
			char *buffer = new char[size + 2];
			buffer[0] = static_cast<uint8_t>(size);
			buffer[0] |= static_cast<uint8_t>(0x80);
			buffer[1] = static_cast<uint8_t>(size >> 7);
			message->SerializeToArray(buffer + 2, size);
			if (send(sock, buffer, size + 2, MSG_NOSIGNAL) < 0)
			{
				reconnect();
			}
			delete[] buffer;
			delete message;
		}

		float get_Euclidean_DistanceAB(float a_x, float a_y, float b_x, float b_y)
		{
			float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
			return std::sqrt(dist);
		}

		bool isThumbNearFirstFinger(NormalizedLandmark point1, NormalizedLandmark point2)
		{
			float distance = this->get_Euclidean_DistanceAB(point1.x(), point1.y(), point2.x(), point2.y());
			return distance < 0.1;
		}

		float getAngleABC(float a_x, float a_y, float b_x, float b_y, float c_x, float c_y)
		{
			float ab_x = b_x - a_x;
			float ab_y = b_y - a_y;
			float cb_x = b_x - c_x;
			float cb_y = b_y - c_y;

			float dot = (ab_x * cb_x + ab_y * cb_y);
			float cross = (ab_x * cb_y - ab_y * cb_x);

			float alpha = std::atan2(cross, dot);

			return alpha;
		}

		int radianToDegree(float radian)
		{
			return (int)floor(radian * 180. / M_PI + 0.5);
		}
	};

	REGISTER_CALCULATOR(HandGestureRecognitionCalculator);

	::mediapipe::Status HandGestureRecognitionCalculator::GetContract(
		CalculatorContract *cc)
	{
		RET_CHECK(cc->Inputs().HasTag(normalizedLandmarkListTag));
		cc->Inputs().Tag(normalizedLandmarkListTag).Set<mediapipe::NormalizedLandmarkList>();

		RET_CHECK(cc->Inputs().HasTag(normRectTag));
		cc->Inputs().Tag(normRectTag).Set<NormalizedRect>();

		RET_CHECK(cc->Inputs().HasTag(landmarkListTag));
		cc->Inputs().Tag(landmarkListTag).Set<mediapipe::LandmarkList>();

		RET_CHECK(cc->Outputs().HasTag(recognizedHandMouvementScrollingTag));
		cc->Outputs().Tag(recognizedHandMouvementScrollingTag).Set<std::string>();

		return ::mediapipe::OkStatus();
	}

	::mediapipe::Status HandGestureRecognitionCalculator::Open(
		CalculatorContext *cc)
	{
		cc->SetOffset(TimestampDiff(0));
		return ::mediapipe::OkStatus();
	}

	::mediapipe::Status HandGestureRecognitionCalculator::Process(
		CalculatorContext *cc)
	{
		Counter *frameCounter = cc->GetCounter("HandGestureRecognitionCalculator");
		frameCounter->Increment();

		const auto rect = &(cc->Inputs().Tag(normRectTag).Get<NormalizedRect>());
		const float height = rect->height();
		const float x_center = rect->x_center();
		const float y_center = rect->y_center();

		if (cc->Inputs().Tag(normalizedLandmarkListTag).IsEmpty())
			return ::mediapipe::OkStatus();
		const auto &landmarkList = cc->Inputs()
									   .Tag(normalizedLandmarkListTag)
									   .Get<mediapipe::NormalizedLandmarkList>();
		RET_CHECK_GT(landmarkList.landmark_size(), 0) << "Input landmark vector is empty.";

		if (cc->Inputs().Tag(landmarkListTag).IsEmpty())
			return ::mediapipe::OkStatus();

		const auto &landmarks = cc->Inputs()
									.Tag(landmarkListTag)
									.Get<mediapipe::LandmarkList>();
		RET_CHECK_GT(landmarks.landmark_size(), 0) << "Input landmark vector is empty.";

		naki3d::common::protocol::SensorDataMessage *data;
		const auto p1 = std::chrono::system_clock::now();

		naki3d::common::protocol::Vector3Data *position = new naki3d::common::protocol::Vector3Data();
		position->set_y(y_center);
		position->set_z(rect->width());
		// position->set_z(landmarkList.landmark(0).z());
		position->set_x(x_center);

		data = new naki3d::common::protocol::SensorDataMessage();
		
		data->set_path("mediapipe/handtracking/hand/right/center_position");
		data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
		data->set_allocated_vector3(position);

		sendMessage(data); // this also frees the memory of data

		// Scrolling
		if (this->previous_x_center)
		{
			std::chrono::duration<double> diff = std::chrono::steady_clock::now() - previous_frame_time;
			std::chrono::duration<double> gestureDiff = std::chrono::steady_clock::now() - previous_gesture_time;
			if (diff < max_frame_time_difference && gestureDiff > min_gesture_time_difference)
			{
				const float movementDistance = this->get_Euclidean_DistanceAB(x_center, y_center, this->previous_x_center, this->previous_y_center);
				// const float movementDistance = this->get_Euclidean_DistanceAB(landmarks.landmark(12).x(), landmarks.landmark(12).y(), this->last_x, this->last_y);

				const float movementDistanceThreshold = movementDistanceFactor * height;

				if (movementDistance > movementDistanceThreshold)
				{
					const float angle = this->radianToDegree(this->getAngleABC(x_center, y_center, this->previous_x_center, this->previous_y_center, this->previous_x_center + 0.1, this->previous_y_center));
					// const float angle = this->radianToDegree(this->getAngleABC(landmarks.landmark(12).x(), landmarks.landmark(12).x(), this->last_x, this->last_y, this->last_x + 0.1, this->last_y));

					if (angle >= -45 && angle < 45)
					{
						LOG(INFO) << "Scrolling right";
						data = new naki3d::common::protocol::SensorDataMessage();
						data->set_path("mediapipe/handtracking/hand/right/gestures/swipe_right");
						data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
						data->set_allocated_void_(new google::protobuf::Empty());
						sendMessage(data); // this also frees the memory of data

						previous_gesture_time = std::chrono::steady_clock::now();
					}
					else if (angle >= 45 && angle < 135)
					{
						LOG(INFO) << "Scrolling up";
						data = new naki3d::common::protocol::SensorDataMessage();
						data->set_path("mediapipe/handtracking/hand/right/gestures/swipe_up");
						data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
						data->set_allocated_void_(new google::protobuf::Empty());
						sendMessage(data); // this also frees the memory of data

						previous_gesture_time = std::chrono::steady_clock::now();
					}
					else if (angle >= 135 || angle < -135)
					{
						LOG(INFO) << "Scrolling left";
						data = new naki3d::common::protocol::SensorDataMessage();
						data->set_path("mediapipe/handtracking/hand/right/gestures/swipe_left");
						data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
						data->set_allocated_void_(new google::protobuf::Empty());
						sendMessage(data); // this also frees the memory of data

						previous_gesture_time = std::chrono::steady_clock::now();
					}
					else if (angle >= -135 && angle < -45)
					{
						LOG(INFO) << "Scrolling down";
						data = new naki3d::common::protocol::SensorDataMessage();
						data->set_path("mediapipe/handtracking/hand/right/gestures/swipe_down");
						data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
						data->set_allocated_void_(new google::protobuf::Empty());
						sendMessage(data); // this also frees the memory of data
						previous_gesture_time = std::chrono::steady_clock::now();
					}
				}
			}
		}
		this->previous_x_center = x_center;
		this->previous_y_center = y_center;
		this->previous_frame_time = std::chrono::steady_clock::now();
		this->last_x = landmarks.landmark(12).x();
		this->last_y = landmarks.landmark(12).y();

		// Finger state
		bool thumbIsOpen = false;
		bool firstFingerIsOpen = false;
		bool secondFingerIsOpen = false;
		bool thirdFingerIsOpen = false;
		bool fourthFingerIsOpen = false;
		int detected_gesture = 0;

		naki3d::common::protocol::Vector3Data *thumbPosition = new naki3d::common::protocol::Vector3Data();
		thumbPosition->set_x(landmarks.landmark(4).x());
		thumbPosition->set_y(landmarks.landmark(4).y());
		thumbPosition->set_z(landmarks.landmark(4).z());

		data = new naki3d::common::protocol::SensorDataMessage();
		data->set_path("mediapipe/handtracking/hand/right/fingers/thumb/position");
		data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
		data->set_allocated_vector3(thumbPosition);
		sendMessage(data); // this also frees the memory of data

		naki3d::common::protocol::Vector3Data *indexPosition = new naki3d::common::protocol::Vector3Data();
		indexPosition->set_x(landmarks.landmark(8).x());
		indexPosition->set_y(landmarks.landmark(8).y());
		indexPosition->set_z(landmarks.landmark(8).z());

		data = new naki3d::common::protocol::SensorDataMessage();
		data->set_path("mediapipe/handtracking/hand/right/fingers/index/position");
		data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
		data->set_allocated_vector3(indexPosition);
		sendMessage(data); // this also frees the memory of data

		naki3d::common::protocol::Vector3Data *middlePosition = new naki3d::common::protocol::Vector3Data();
		middlePosition->set_x(landmarks.landmark(12).x());
		middlePosition->set_y(landmarks.landmark(12).y());
		middlePosition->set_z(landmarks.landmark(12).z());

		data = new naki3d::common::protocol::SensorDataMessage();
		data->set_path("mediapipe/handtracking/hand/right/fingers/middle/position");
		data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
		data->set_allocated_vector3(middlePosition);
		sendMessage(data); // this also frees the memory of data

		naki3d::common::protocol::Vector3Data *ringPosition = new naki3d::common::protocol::Vector3Data();
		ringPosition->set_x(landmarks.landmark(16).x());
		ringPosition->set_y(landmarks.landmark(16).y());
		ringPosition->set_z(landmarks.landmark(16).z());

		data = new naki3d::common::protocol::SensorDataMessage();
		data->set_path("mediapipe/handtracking/hand/right/fingers/ring/position");
		data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
		data->set_allocated_vector3(ringPosition);
		sendMessage(data); // this also frees the memory of data

		naki3d::common::protocol::Vector3Data *pinkyPosition = new naki3d::common::protocol::Vector3Data();
		pinkyPosition->set_x(landmarks.landmark(20).x());
		pinkyPosition->set_y(landmarks.landmark(20).y());
		pinkyPosition->set_z(landmarks.landmark(20).z());

		data = new naki3d::common::protocol::SensorDataMessage();
		data->set_path("mediapipe/handtracking/hand/right/fingers/pinky/position");
		data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
		data->set_allocated_vector3(pinkyPosition);
		sendMessage(data); // this also frees the memory of data

		float pseudoFixKeyPoint = landmarkList.landmark(2).x();
		if (landmarkList.landmark(3).x() < pseudoFixKeyPoint && landmarkList.landmark(4).x() < pseudoFixKeyPoint)
		{
			thumbIsOpen = true;
			data = new naki3d::common::protocol::SensorDataMessage();
			data->set_path("mediapipe/handtracking/hand/right/fingers/thumb/closed");
			data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
			data->set_bool_(false);
			sendMessage(data); // this also frees the memory of data
		}

		pseudoFixKeyPoint = landmarkList.landmark(6).y();
		if (landmarkList.landmark(7).y() < pseudoFixKeyPoint && landmarkList.landmark(8).y() < pseudoFixKeyPoint)
		{
			firstFingerIsOpen = true;
			data = new naki3d::common::protocol::SensorDataMessage();
			data->set_path("mediapipe/handtracking/hand/right/fingers/index/closed");
			data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
			data->set_bool_(false);
			sendMessage(data); // this also frees the memory of data
		}

		pseudoFixKeyPoint = landmarkList.landmark(10).y();
		if (landmarkList.landmark(11).y() < pseudoFixKeyPoint && landmarkList.landmark(12).y() < pseudoFixKeyPoint)
		{
			secondFingerIsOpen = true;
			data = new naki3d::common::protocol::SensorDataMessage();
			data->set_path("mediapipe/handtracking/hand/right/fingers/middle/closed");
			data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
			data->set_bool_(false);
			sendMessage(data); // this also frees the memory of data
		}

		pseudoFixKeyPoint = landmarkList.landmark(14).y();
		if (landmarkList.landmark(15).y() < pseudoFixKeyPoint && landmarkList.landmark(16).y() < pseudoFixKeyPoint)
		{
			thirdFingerIsOpen = true;
			data = new naki3d::common::protocol::SensorDataMessage();
			data->set_path("mediapipe/handtracking/hand/right/fingers/ring/closed");
			data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
			data->set_bool_(false);
			sendMessage(data); // this also frees the memory of data
		}

		pseudoFixKeyPoint = landmarkList.landmark(18).y();
		if (landmarkList.landmark(19).y() < pseudoFixKeyPoint && landmarkList.landmark(20).y() < pseudoFixKeyPoint)
		{
			fourthFingerIsOpen = true;
			data = new naki3d::common::protocol::SensorDataMessage();
			data->set_path("mediapipe/handtracking/hand/right/fingers/pinky/closed");
			data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
			data->set_bool_(false);
			sendMessage(data); // this also frees the memory of data
		}

		if (!firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
		{
			detected_gesture = 0;
		}
		else if (get_Euclidean_DistanceAB(landmarkList.landmark(4).x(), landmarkList.landmark(4).y(), landmarkList.landmark(8).x(), landmarkList.landmark(8).y()) <= thumb_index_distance_threshold &&
				 get_Euclidean_DistanceAB(landmarkList.landmark(12).x(), landmarkList.landmark(12).y(), landmarkList.landmark(8).x(), landmarkList.landmark(8).y()) >= index_middle_distance_threshold)
		{
			detected_gesture = 1;
		}
		else
		{
			detected_gesture = 2;
		}

		if (detected_gesture != last_gesture)
		{
			data = new naki3d::common::protocol::SensorDataMessage();
			const auto p1 = std::chrono::system_clock::now();

			switch (detected_gesture)
			{
			case 0:
				LOG(INFO) << "Closed Hand";
				data->set_path("mediapipe/handtracking/hand/right/gestures/close_hand");
				data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
				data->set_allocated_void_(new google::protobuf::Empty());
				sendMessage(data); // this also frees the memory of data
				break;
			case 1:
				LOG(INFO) << "Pinch";
				data->set_path("mediapipe/handtracking/hand/right/gestures/pinch");
				data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
				data->set_allocated_void_(new google::protobuf::Empty());
				sendMessage(data); // this also frees the memory of data
				break;
			case 2:
				LOG(INFO) << "Open Hand";
				data->set_path("mediapipe/handtracking/hand/right/gestures/open_hand");
				data->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
				data->set_allocated_void_(new google::protobuf::Empty());
				sendMessage(data); // this also frees the memory of data
				break;
			}

			last_gesture = detected_gesture;
		}

		// TODO: does this need to be here?
		cc->Outputs()
			.Tag(recognizedHandMouvementScrollingTag)
			.Add(new std::string("___"), cc->InputTimestamp());

		return ::mediapipe::OkStatus();
	}

}
