css = """
<style>
    .chat-container {
    max-height: 65vh; /* Less than full screen */
    overflow-y: auto;
    padding: 10px 20px 120px; /* bottom padding = input box space */
    display: flex;
    flex-direction: column;
    gap: 16px;
    scroll-behavior: smooth;
    }
    .user-msg, .bot-msg {
        padding: 1rem;
        border-radius: 10px;
        max-width: 80%;
    }
    .user-msg {
        background-color: #1c1c1c;
        color: white;
        align-self: flex-end;
    }
    .bot-msg {
        background-color: #f1f1f1;
        color: black;
        align-self: flex-start;
    }
</style>
"""


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        🤖
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        🙋‍♂️
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
