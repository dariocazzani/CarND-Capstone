

import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight


ssd_inception_sim_model  = 'models/real/frozen_inference_graph.pb'


class TLClassifier(object,name):
    def __init__(self):
        #TODO load classifier
        self.graph=tf.Graph()
        
        with detection_graph.as_default():
            od_graph_def= tf.GraphDef()
            with tf.gfile.GFile(ssd_inception_sim_model,'rb') as fid:
                serialized_graph= fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
            self.image_tensor=detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_scores=detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes=detection_graph.get_tensor_by_name('detection_classes:0')
            
            self.sess= tf.Session(graph=self.graph)
            
        
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        calss_vote_green=0
        class_vote_red=0
        class_vote_yellow=0
        class_threshold =.85
        
        with self.graph.as_default():
            image_np_expanded= np.expand_dims(image,axis=0)
            
            (scores, classes)= sess.run([detection_scores,detection_classes],
                                        feed_dict={image_tensor:image_np_expanded})
            scores=np.squeeze(scores)
            classes=np.squeeze(classes)
            classes=classes.astype(int)
            for i in range(scores):
                if scores[i]>class_threshold:
                    if classes[i]==1:
                        class_vote_green+=1
                    elif classes[i]==2:
                        class_vote_red+=1
                    elif classes[i]==3:
                        class_vote_yellow+=1
                        
            if class_vote_green>0:
                return TrafficLight.GREEN
            elif class_vote_red>0:
                return TrafficLight.RED
            elif class_vote_yellow>0:
                return TrafficLight.YELLOW
            
            return TrafficLight.UNKNOWN