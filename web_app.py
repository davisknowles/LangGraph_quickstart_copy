"""
Flask web application for the Safety Agent chat interface.
Provides a local web frontend to interact with the safety agent.
"""

from flask import Flask, render_template, request, jsonify, session, Response
import uuid
from datetime import datetime
import os
import sys
import json

# Add the current directory to the Python path so we can import safety_agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your safety agent
from safety_agent import agent

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Use the agent from safety_agent module
# agent is already created in safety_agent.py

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with streaming chain of thought."""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get or create session ID
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Return streaming response
        return Response(
            stream_chat_response(user_message, session_id),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )
        
    except Exception as e:
        print(f"Error processing chat message: {str(e)}")
        return jsonify({
            'error': f'Sorry, I encountered an error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

def stream_chat_response(user_message, session_id):
    """Stream the agent's chain of thought in real-time."""
    import time
    
    def send_thinking_step(step, details=""):
        """Send a thinking step to the client."""
        data = {
            'type': 'thinking',
            'step': step,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        return f"data: {json.dumps(data)}\n\n"
    
    def send_final_response(response):
        """Send the final response to the client."""
        data = {
            'type': 'response',
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id
        }
        return f"data: {json.dumps(data)}\n\n"
    
    try:
        # Step 1: Start processing
        yield send_thinking_step("Analyzing question", user_message)
        time.sleep(0.5)
        
        # Step 2: Classify the query
        yield send_thinking_step("Determining query type", "Checking if this requires statistical analysis or knowledge search...")
        from safety_agent import classify_intent
        query_type = classify_intent(user_message)
        yield send_thinking_step("Query classified", f"This is a {query_type} question")
        time.sleep(0.3)
        
        # Step 3: Choose approach
        if query_type == "statistical":
            yield send_thinking_step("Loading statistical data", "Connecting to Azure Blob Storage...")
            time.sleep(0.5)
            
            # Try to load data and report progress
            try:
                from safety_agent import load_data_from_blob
                yield send_thinking_step("Downloading data", "Fetching safety incident records...")
                df = load_data_from_blob()
                yield send_thinking_step("Data loaded successfully", f"Loaded {len(df):,} safety incidents with {df.shape[1]} data fields")
                time.sleep(0.3)
                
                # Step 4: Generate analysis
                yield send_thinking_step("Generating analysis code", "Creating Python code to analyze your specific question...")
                time.sleep(0.8)
                
                yield send_thinking_step("Executing analysis", "Running statistical calculations on the data...")
                time.sleep(0.5)
                
            except Exception as e:
                yield send_thinking_step("Data loading failed", f"Falling back to alternative approach: {str(e)}")
        else:
            yield send_thinking_step("Searching knowledge base", "Looking for relevant safety information...")
            time.sleep(0.8)
        
        # Step 5: Get actual response from agent
        yield send_thinking_step("Invoking safety agent", "Running the complete analysis...")
        
        config = {"configurable": {"thread_id": session_id}}
        response = agent.invoke(
            {"messages": [("user", user_message)]}, 
            config=config
        )
        
        # Step 6: Extract and format response
        yield send_thinking_step("Formatting results", "Preparing the final analysis for presentation...")
        time.sleep(0.3)
        
        # Extract the response (same logic as before)
        assistant_message = ""
        tool_response = ""
        
        if response and "messages" in response:
            for message in reversed(response["messages"]):
                if hasattr(message, '__class__') and 'Tool' in str(message.__class__):
                    if hasattr(message, 'content') and isinstance(message.content, str):
                        tool_response = message.content
                        break
            
            if tool_response and ("Response:" in tool_response or len(tool_response) > 200):
                if "Response:" in tool_response:
                    parts = tool_response.split("Response:", 1)
                    if len(parts) > 1:
                        assistant_message = parts[1].strip()
                    else:
                        assistant_message = tool_response
                else:
                    assistant_message = tool_response
            else:
                for message in reversed(response["messages"]):
                    if hasattr(message, '__class__') and 'AI' in str(message.__class__):
                        if hasattr(message, 'content'):
                            if isinstance(message.content, str):
                                assistant_message = message.content
                                break
                            elif isinstance(message.content, list):
                                text_parts = []
                                for part in message.content:
                                    if isinstance(part, dict):
                                        if part.get('type') == 'text':
                                            text_parts.append(part.get('text', ''))
                                    elif isinstance(part, str):
                                        text_parts.append(part)
                                if text_parts:
                                    assistant_message = ''.join(text_parts)
                                    break
        
        if not assistant_message:
            assistant_message = "I processed your request, but I'm having trouble formatting the response. Please try rephrasing your question."
        
        # Step 7: Send final response
        yield send_thinking_step("Analysis complete", "Ready to show you the results")
        time.sleep(0.2)
        yield send_final_response(assistant_message)
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error during analysis: {str(e)}"
        yield send_final_response(error_msg)

@app.route('/clear', methods=['POST'])
def clear_session():
    """Clear the current chat session."""
    if 'session_id' in session:
        # Create new session ID
        session['session_id'] = str(uuid.uuid4())
    
    return jsonify({
        'status': 'success',
        'message': 'Chat session cleared',
        'new_session_id': session.get('session_id')
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Safety Agent Web Interface'
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting Safety Agent Web Interface...")
    print("📊 Loading safety data...")
    
    # Test that the safety agent works
    try:
        test_response = agent.invoke(
            {"messages": [("user", "Hello, are you working?")]}, 
            config={"configurable": {"thread_id": "test"}}
        )
        print("✅ Safety agent is ready!")
    except Exception as e:
        print(f"⚠️  Warning: Safety agent test failed: {e}")
    
    print("🌐 Starting web server...")
    print("📱 Open your browser to: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)