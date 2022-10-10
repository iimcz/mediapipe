#include <iostream>
#include "wrappers.pb.h"
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#define PORT 8080

int main(){
	int sock = 0, valread, client_fd;
  struct sockaddr_in serv_addr;
  char* hello = "Hello from client";
  char buffer[1024] = { 0 };
  if((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0){
  	printf("\n Socket creation error \n");
   	return -1;
  }		
 
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(PORT);

	if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0){
        printf("\nInvalid address\n");
        return -1;
  }
 
  if((client_fd = connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr))) < 0) {
  	printf("\nConnection Failed \n");
    return -1;
  }
  send(sock, hello, strlen(hello), 0);
	
	const std::string ID = "rpi-cammera";
	naki3d::common::protocol::SensorMessage message;
	message.set_sensor_id(ID);
	message.set_timestamp(32315151313);
	naki3d::common::protocol::MediapipeHandTrackingData data;
	data.set_side(naki3d::common::protocol::HandSide::LEFT);
	std::cout << "a" << std::endl;
	naki3d::common::protocol::Vector3 position;
	position.set_x(1);
	position.set_y(1);
	position.set_z(1);
	data.set_gesture(naki3d::common::protocol::HandGestureType::GESTURE_SWIPE_LEFT);
	return 0;
}
