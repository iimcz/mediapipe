#include <cmath>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

#include "protobuf/wrappers.pb.h"

namespace mediapipe
{

    namespace
    {
        constexpr char normRectTag[] = "NORM_RECT";
        constexpr char normalizedLandmarkListTag[] = "NORM_LANDMARKS";
        constexpr char recognizedHandMouvementScrollingTag[] = "RECOGNIZED_HAND_GESTURE";
    }

    class HandGestureRecognitionCalculator : public CalculatorBase
    {
    public:
        static ::mediapipe::Status GetContract(CalculatorContract *cc);
        ::mediapipe::Status Open(CalculatorContext *cc) override;

        ::mediapipe::Status Process(CalculatorContext *cc) override;

    private:
        float previous_x_center;
        float previous_y_center;
        float previous_angle;
        float previous_rectangle_width;
        float previous_rectangle_height;

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

        std::string *recognized_hand_movement_scrolling = new std::string("___");
        std::string *recognized_hand_movement_zooming = new std::string("___");

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

        //Scrolling
        if (this->previous_x_center)
        {
            const float movementDistance = this->get_Euclidean_DistanceAB(x_center, y_center, this->previous_x_center, this->previous_y_center);

            const float movementDistanceFactor = 0.3; //movement threshold.

            const float movementDistanceThreshold = movementDistanceFactor * height;
            
            const std::string ID = "rpi-cammera";
						naki3d::common::protocol::SensorMessage message;
						message.set_sensor_id(ID);
						message.set_timestamp(32315151313);
						naki3d::common::protocol::MediapipeHandTrackingData data;
						data.set_side(naki3d::common::protocol::HandSide::LEFT);
            
            if (movementDistance > movementDistanceThreshold)
            {
                const float angle = this->radianToDegree(this->getAngleABC(x_center, y_center, this->previous_x_center, this->previous_y_center, this->previous_x_center + 0.1, this->previous_y_center));
                if (angle >= -45 && angle < 45)
                {
                    recognized_hand_movement_scrolling = new std::string("Scrolling right");
                    data.set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_SWIPE_RIGHT);
                }
                else if (angle >= 45 && angle < 135)
                {
                    recognized_hand_movement_scrolling = new std::string("Scrolling up");
                    data.set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_SWIPE_UP);
                }                }
                else if (angle >= 135 || angle < -135)
                {
                    recognized_hand_movement_scrolling = new std::string("Scrolling left");
                    data.set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_SWIPE_LEFT);
                }
                else if (angle >= -135 && angle < -45)
                {
                    recognized_hand_movement_scrolling = new std::string("Scrolling down");
                    data.set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_SWIPE_DOWN);
                }
            }
        }
        this->previous_x_center = x_center;
        this->previous_y_center = y_center;
        
        //Zooming
        if (this->previous_rectangle_height)
        {
            const float heightDifferenceFactor = 0.03;

            const float heightDifferenceThreshold = height * heightDifferenceFactor;
            if (height < this->previous_rectangle_height - heightDifferenceThreshold)
            {
                recognized_hand_movement_zooming = new std::string("Zoom out");
            }
            else if (height > this->previous_rectangle_height + heightDifferenceThreshold)
            {
                recognized_hand_movement_zooming = new std::string("Zoom in");
            }
        }
        this->previous_rectangle_height = height;

        LOG(INFO) << recognized_hand_movement_scrolling->c_str();
        LOG(INFO) << recognized_hand_movement_zooming->c_str();

        cc->Outputs()
            .Tag(recognizedHandMouvementScrollingTag)
            .Add(recognized_hand_movement_scrolling, cc->InputTimestamp());

        return ::mediapipe::OkStatus();
    }

}
