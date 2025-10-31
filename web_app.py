"""
Flask web application for the Safety Agent chat interface.
Provides a local web frontend to interact with the safety agent.
"""

from flask import Flask, render_template, request, jsonify, session
import uuid
from datetime import datetime
import os
import sys

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
    """Handle chat messages."""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get or create session ID
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Create the configuration for the agent
        config = {"configurable": {"thread_id": session_id}}
        
        # Get response from the agent
        print(f"Processing message: {user_message}")
        
        # Invoke the agent
        response = agent.invoke(
            {"messages": [("user", user_message)]}, 
            config=config
        )
        
        # Debug: Print response structure
        print(f"Response keys: {response.keys() if response else 'None'}")
        if response and "messages" in response:
            print(f"Number of messages: {len(response['messages'])}")
            for i, msg in enumerate(response["messages"]):
                print(f"Message {i}: {type(msg).__name__}")
                if hasattr(msg, 'content'):
                    content_preview = str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content)
                    print(f"  Content preview: {content_preview}")
        
        # Extract the assistant's response from LangGraph format
        assistant_message = ""
        tool_response = ""
        
        if response and "messages" in response:
            # First, look for ToolMessage responses (these contain the actual analysis)
            for message in reversed(response["messages"]):
                if hasattr(message, '__class__') and 'Tool' in str(message.__class__):
                    if hasattr(message, 'content') and isinstance(message.content, str):
                        tool_response = message.content
                        break
            
            # If we found a tool response with detailed content, use it
            if tool_response and ("Response:" in tool_response or len(tool_response) > 200):
                # Extract just the response part if it has the full format
                if "Response:" in tool_response:
                    parts = tool_response.split("Response:", 1)
                    if len(parts) > 1:
                        assistant_message = parts[1].strip()
                    else:
                        assistant_message = tool_response
                else:
                    assistant_message = tool_response
            else:
                # Otherwise, look for AI message
                for message in reversed(response["messages"]):
                    if hasattr(message, '__class__') and 'AI' in str(message.__class__):
                        if hasattr(message, 'content'):
                            if isinstance(message.content, str):
                                assistant_message = message.content
                                break
                            elif isinstance(message.content, list):
                                # Extract text from structured content
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
        
        # Fallback response if we couldn't extract anything
        if not assistant_message:
            assistant_message = "I processed your request, but I'm having trouble formatting the response. Please try rephrasing your question."
        
        return jsonify({
            'response': assistant_message,
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error processing chat message: {str(e)}")
        return jsonify({
            'error': f'Sorry, I encountered an error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

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