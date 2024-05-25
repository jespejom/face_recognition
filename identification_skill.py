#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy 
from uchile_skills.robot_skill import RobotSkill
from std_msgs.msg import String
import math
import recognition_node as rn
import detection_node as dn

class IdentificationSkill(RobotSkill):    
    def __init__(self):
        super(IdentificationSkill, self).__init__()
        self._type = "IdentificationSkill"
        self._name = "identification_skill"
        self._description = "Skill to identify people"

        #self.last_identification = None
    
    def setup(self):
        self.detector = dn.face_detection_node()
        self.recognizer = rn.face_recognition_node()
        return True
    
    def start(self):
        self.loginfo("{skill: %s}: start()." % self._type)
        try:
            self.detector.start_callback()
            self.recognizer.start_callback()
        except Exception, e:
            raise e
        return True

    def shutdown(self):
        self.loginfo("Shutting Down")
        try:
            self.detector.stop_callback()
            self.recognizer.stop_callback()
        except Exception, e:
            raise e
        return True
       
    def check(self):
        return True

    def stop(self):
        self.loginfo("{skill: %s}: stop()." % self._type)
        try:
            self.detector.stop_callback()
            self.recognizer.stop_callback()
        except Exception, e:
            raise e
        return True
    
    def save_face(self, face, name):
        # Falta implementar codigo que interactue con la conversación 
        # y pueda entregar a este metodo el nombre (si corresponde) 
        # y la autorización para guardar la imagen
        self.recognizer.add_face_to_facebank(face, name)
        return True