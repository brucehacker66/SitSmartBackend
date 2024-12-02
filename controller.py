from flask import Flask, request, jsonify
from threading import Thread, Event
import service
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Endpoint to get the current posture status for a user
@app.route('/sitsmart/api/status/<user_id>', methods=['GET'])
def get_status(user_id):
    status = service.get_posture_status(user_id)
    
    if status == "Processing":
        return jsonify({"status": "Processing"}), 200
    
    return jsonify({"status": status}), 200

# Endpoint to set the interval for a specific user
@app.route('/sitsmart/api/interval/<user_id>', methods=['POST'])
def set_interval(user_id):
    data = request.get_json()
    interval = data.get('interval')
    
    if not interval or not isinstance(interval, int) or interval <= 0:
        return jsonify({"error": "Invalid interval provided"}), 400
    
    service.set_user_interval(user_id, interval)
    return jsonify({"message": f"Interval for user {user_id} set to {interval} seconds"}), 200

# Endpoint to add a new user and start their status update thread
@app.route('/sitsmart/api/add_user/<user_id>', methods=['POST'])
def add_user(user_id):
    if service.user_exists(user_id):
        return jsonify({"error": "User already exists"}), 400
    
    service.add_user(user_id)
    return jsonify({"message": f"User {user_id} added with default interval {service.default_interval} seconds"}), 200

# Endpoint to delete a user and stop their status update thread
@app.route('/sitsmart/api/delete_user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    if not service.user_exists(user_id):
        return jsonify({"error": "User does not exist"}), 400
    
    service.delete_user(user_id)
    return jsonify({"message": f"User {user_id} deleted"}), 200

# Endpoint to update the directory for a user
@app.route('/sitsmart/api/update_directory/<user_id>', methods=['POST'])
def update_directory(user_id):
    if not service.user_exists(user_id):
        return jsonify({"error": "User does not exist"}), 400
    
    data = request.get_json()
    directory_path = data.get("directoryPath")
    
    if not directory_path:
        return jsonify({"error": "Directory path is required"}), 400

    message = service.update_user_directory(user_id, directory_path)
    return jsonify({"message": message}), 200

# Run the Flask app on localhost:5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)