from flask import Flask, request, jsonify
from threading import Thread
import service

app = Flask(__name__)

# Start the worker thread
worker_thread = Thread(target=service.process_queue, daemon=True)
worker_thread.start()

# Endpoint to add image ID to the processing queue
# image_id should map the id in the database
@app.route('sitsmart/api/infer', methods=['POST'])
def infer():
    data = request.get_json()
    user_id = data.get('user_id')
    image_id = data.get('image_id')

    if not user_id or not image_id:
        return jsonify({"error": "user_id and image_id are required"}), 400

    # Add image ID to the queue using service logic
    service.add_to_queue(user_id, image_id)

    return jsonify({"message": "Image ID added to processing queue"}), 202

# Endpoint to get the current posture status for a user
@app.route('sitsmart/api/status/<user_id>', methods=['GET'])
def get_status(user_id):
    status = service.get_posture_status(user_id)
    
    if status == "Processing":
        return jsonify({"status": "Processing"}), 200
    
    return jsonify(status), 200

# the app will run on localhost:5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)