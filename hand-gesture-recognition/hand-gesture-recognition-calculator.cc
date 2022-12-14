#include <cmath>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

#include "protobuf/wrappers.pb.h"

#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <thread>

#define PORT 5000
#define IP "192.168.22.172"

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

		int last_gesture = 0;

		const float thumb_index_distance_threshold = 0.03;
		const float index_middle_distance_threshold = 0.06;
		const float movementDistanceFactor = 0.15; // movement threshold.
		std::chrono::duration<double> max_frame_time_difference = std::chrono::milliseconds(100);

		std::chrono::time_point<std::chrono::steady_clock> previous_frame_time;

		int sock = 0, client_fd;
		struct sockaddr_in serv_addr;

		using Time = std::chrono::steady_clock;
		using float_sec = std::chrono::duration<float>;
		using float_time_point = std::chrono::time_point<Time, float_sec>;

		bool reconnecting = false;
		float_time_point reconnect_start;

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
				reconnecting = false;
			}
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

		naki3d::common::protocol::Vector3 *position = new naki3d::common::protocol::Vector3();
		position->set_y(y_center);
		position->set_z(rect->width());
		// position->set_z(landmarkList.landmark(0).z());
		position->set_x(x_center);

		naki3d::common::protocol::MediapipeHandTrackingData *data = new naki3d::common::protocol::MediapipeHandTrackingData();
		data->set_side(naki3d::common::protocol::HandSide::RIGHT);
		data->set_allocated_center_position(position);
		data->set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_NONE);

		// Scrolling
		if (this->previous_x_center)
		{
			std::chrono::duration<double> diff = std::chrono::steady_clock::now() - previous_frame_time;
			if (diff < max_frame_time_difference)
			{
				const float movementDistance = this->get_Euclidean_DistanceAB(x_center, y_center, this->previous_x_center, this->previous_y_center);
				const float movementDistanceThreshold = movementDistanceFactor * height;

				if (movementDistance > movementDistanceThreshold)
				{
					const float angle = this->radianToDegree(this->getAngleABC(x_center, y_center, this->previous_x_center, this->previous_y_center, this->previous_x_center + 0.1, this->previous_y_center));
					if (angle >= -45 && angle < 45)
					{
						LOG(INFO) << "Scrolling right";
						data->set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_SWIPE_RIGHT);
					}
					else if (angle >= 45 && angle < 135)
					{
						LOG(INFO) << "Scrolling up";
						data->set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_SWIPE_UP);
					}
					else if (angle >= 135 || angle < -135)
					{
						LOG(INFO) << "Scrolling left";
						data->set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_SWIPE_LEFT);
					}
					else if (angle >= -135 && angle < -45)
					{
						LOG(INFO) << "Scrolling down";
						data->set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_SWIPE_DOWN);
					}
				}
			}
		}
		this->previous_x_center = x_center;
		this->previous_y_center = y_center;
		this->previous_frame_time = std::chrono::steady_clock::now();

		// Finger state
		bool thumbIsOpen = false;
		bool firstFingerIsOpen = false;
		bool secondFingerIsOpen = false;
		bool thirdFingerIsOpen = false;
		bool fourthFingerIsOpen = false;
		int detected_gesture = 0;

		naki3d::common::protocol::Vector3 *thumbPosition = new naki3d::common::protocol::Vector3();
		thumbPosition->set_x(landmarks.landmark(4).x());
		thumbPosition->set_y(landmarks.landmark(4).y());
		thumbPosition->set_z(landmarks.landmark(4).z());

		naki3d::common::protocol::FingerState *thumb = new naki3d::common::protocol::FingerState();
		thumb->set_closed(true);
		thumb->set_allocated_position(thumbPosition);

		naki3d::common::protocol::Vector3 *indexPosition = new naki3d::common::protocol::Vector3();
		indexPosition->set_x(landmarks.landmark(8).x());
		indexPosition->set_y(landmarks.landmark(8).y());
		indexPosition->set_z(landmarks.landmark(8).z());

		naki3d::common::protocol::FingerState *index = new naki3d::common::protocol::FingerState();
		index->set_closed(true);
		index->set_allocated_position(indexPosition);

		naki3d::common::protocol::Vector3 *middlePosition = new naki3d::common::protocol::Vector3();
		middlePosition->set_x(landmarks.landmark(12).x());
		middlePosition->set_y(landmarks.landmark(12).y());
		middlePosition->set_z(landmarks.landmark(12).z());

		naki3d::common::protocol::FingerState *middle = new naki3d::common::protocol::FingerState();
		middle->set_closed(true);
		middle->set_allocated_position(middlePosition);

		naki3d::common::protocol::Vector3 *ringPosition = new naki3d::common::protocol::Vector3();
		ringPosition->set_x(landmarks.landmark(16).x());
		ringPosition->set_y(landmarks.landmark(16).y());
		ringPosition->set_z(landmarks.landmark(16).z());

		naki3d::common::protocol::FingerState *ring = new naki3d::common::protocol::FingerState();
		ring->set_closed(true);
		ring->set_allocated_position(ringPosition);

		naki3d::common::protocol::Vector3 *pinkyPosition = new naki3d::common::protocol::Vector3();
		pinkyPosition->set_x(landmarks.landmark(20).x());
		pinkyPosition->set_y(landmarks.landmark(20).y());
		pinkyPosition->set_z(landmarks.landmark(20).z());

		naki3d::common::protocol::FingerState *pinky = new naki3d::common::protocol::FingerState();
		pinky->set_closed(true);
		pinky->set_allocated_position(pinkyPosition);

		float pseudoFixKeyPoint = landmarkList.landmark(2).x();
		if (landmarkList.landmark(3).x() < pseudoFixKeyPoint && landmarkList.landmark(4).x() < pseudoFixKeyPoint)
		{
			thumbIsOpen = true;
			thumb->set_closed(false);
		}

		pseudoFixKeyPoint = landmarkList.landmark(6).y();
		if (landmarkList.landmark(7).y() < pseudoFixKeyPoint && landmarkList.landmark(8).y() < pseudoFixKeyPoint)
		{
			firstFingerIsOpen = true;
			index->set_closed(false);
		}

		pseudoFixKeyPoint = landmarkList.landmark(10).y();
		if (landmarkList.landmark(11).y() < pseudoFixKeyPoint && landmarkList.landmark(12).y() < pseudoFixKeyPoint)
		{
			secondFingerIsOpen = true;
			middle->set_closed(false);
		}

		pseudoFixKeyPoint = landmarkList.landmark(14).y();
		if (landmarkList.landmark(15).y() < pseudoFixKeyPoint && landmarkList.landmark(16).y() < pseudoFixKeyPoint)
		{
			thirdFingerIsOpen = true;
			ring->set_closed(false);
		}

		pseudoFixKeyPoint = landmarkList.landmark(18).y();
		if (landmarkList.landmark(19).y() < pseudoFixKeyPoint && landmarkList.landmark(20).y() < pseudoFixKeyPoint)
		{
			fourthFingerIsOpen = true;
			pinky->set_closed(false);
		}

		naki3d::common::protocol::HandFingerState *fingerState = new naki3d::common::protocol::HandFingerState();
		fingerState->set_allocated_thumb(thumb);
		fingerState->set_allocated_index(index);
		fingerState->set_allocated_middle(middle);
		fingerState->set_allocated_ring(ring);
		fingerState->set_allocated_pinky(pinky);

		data->set_allocated_finger_state(fingerState);

		naki3d::common::protocol::SensorMessage *message = new naki3d::common::protocol::SensorMessage();
		message->set_sensor_id("rpi-cammera");
		const auto p1 = std::chrono::system_clock::now();
		message->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());
		message->set_allocated_handtracking(data);

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
			switch (detected_gesture)
			{
			case 0:
				LOG(INFO) << "Closed Hand";
				data->set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_CLOSE_HAND);
				break;
			case 1:
				LOG(INFO) << "Pinch";
				data->set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_PINCH);
				break;
			case 2:
				LOG(INFO) << "Open Hand";
				data->set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_OPEN_HAND);
				break;
			}

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

			last_gesture = detected_gesture;
		}

		delete message;

		// TODO: does this need to be here?
		cc->Outputs()
			.Tag(recognizedHandMouvementScrollingTag)
			.Add(new std::string("___"), cc->InputTimestamp());

		return ::mediapipe::OkStatus();
	}

}
