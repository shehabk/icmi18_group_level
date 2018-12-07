#!/usr/bin/python
import os
import cv2
import numpy as np
import math

from collections import OrderedDict
class FaceAlignerMTCNN:
    def __init__(self, desiredRightEye=(0.34, 0.35), desiredNoseTipY = 0.6,
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height

        self.desiredRightEye = desiredRightEye
        self.desiredNoseTipY = desiredNoseTipY
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth



    def align(self, image, shape):

        inPts = []
        outPts = []


        # Prepare inPts
        leftEyeCenter  = np.array([ int(shape[1]) , int(shape[1+5]) ])
        rightEyeCenter = np.array([ int(shape[0]) , int(shape[0+5])])


        inPts.append(leftEyeCenter)
        inPts.append(rightEyeCenter)



        # Prepare outPts
        rightEyeCenterOut = np.array([ int(self.desiredRightEye[0]*self.desiredFaceWidth) ,\
                              int(self.desiredRightEye[1]*self.desiredFaceWidth)])

        desiredRightEyeX  = self.desiredRightEye[0]
        desiredLeftEyeX   =  1.0 - desiredRightEyeX

        leftEyeCenterOut = np.array([int(desiredLeftEyeX*self.desiredFaceWidth), \
                            int(self.desiredRightEye[1] * self.desiredFaceWidth)])


        outPts.append(leftEyeCenterOut)
        outPts.append(rightEyeCenterOut)





        ###### For Three point allignment reauires affine to be true
        noseTip = np.array([int(shape[2]), int(shape[2 + 5])])
        noseTipOut = self.getNoseTip(rightEyeCenter, leftEyeCenter, noseTip)
        inPts.append(noseTip)
        outPts.append(noseTipOut)
        M = cv2.estimateRigidTransform(np.array(inPts), np.array(outPts), True);



        # apply the affine transformation
        # M = self.similarityTransform(inPts, outPts)
        if M is None:
            return None

        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h))

        # return the aligned face
        return output

    def similarityTransform(self , inPoints, outPoints):
        s60 = math.sin(60 * math.pi / 180);
        c60 = math.cos(60 * math.pi / 180);

        inPts = np.copy(inPoints).tolist();
        outPts = np.copy(outPoints).tolist();

        xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0];
        yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1];

        inPts.append([np.int(xin), np.int(yin)]);

        xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0];
        yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1];

        outPts.append([np.int(xout), np.int(yout)]);

        tform = cv2.estimateRigidTransform(np.array(inPts), np.array(outPts), False);

        return tform;

    def getNoseTip(self , r , l , n ):

        rl = l - r
        rl_unit = rl / np.linalg.norm(rl)
        rl_norm = np.linalg.norm(rl)

        rn = n - r
        rn_on_rl = np.dot(rl, rn) / rl_norm
        rn_on_rl = r + rn_on_rl * rl_unit
        rn_on_rl_ratio = np.linalg.norm(rn_on_rl -r ) / rl_norm


        dr = np.array( [ self.desiredRightEye[0] , self.desiredRightEye[1]   ] )
        dl = np.array( [1 -self.desiredRightEye[0] , self.desiredRightEye[1] ] )

        d_rn_on_rl = dr + (dl-dr)*rn_on_rl_ratio
        dn = np.array( [d_rn_on_rl[0] , self.desiredNoseTipY] )

        dn_ret = dn*self.desiredFaceWidth
        dn_ret = dn_ret.astype('int')

        return dn_ret

