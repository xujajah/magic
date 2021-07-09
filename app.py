#! /usr/bin/env python

from flask import Flask, request
from flask_restful import Resource, Api
import magic

app = Flask(__name__)
api = Api(app)

class MagicMeasurements(Resource):
    def post(self):
        data = None
        if request.json:
            data = request.json
        elif request.values:
            data = request.values
        else:
            data = request.data

        if not data:
            return {
                "success": False,
                "error": "failed to retrieve required image locations"
            }, 400
        
        calib, front, side = data.get('calib'), data.get('front'), data.get('side')

        if not calib or not front or not side:
            return {
                "success": False,
                "error": "failed to retrieve required image locations"
            }, 400

        status, result = magic.magicMeasurements(calib, front, side)
        if not status:
            if result == magic.READ_ERROR:
                return {
                        "success": False,
                        "error": "failed to read images"
                    }, 500
            elif result == magic.PROCESS_ERROR:
                return {
                        "success": False,
                        "error": "failed to process images"
                    }, 500
            else:
                return {
                        "success": False,
                        "error": "unexpected error"
                    }, 500
        
        result['success'] = True
        return result, 200

api.add_resource(MagicMeasurements, "/magic/api/")