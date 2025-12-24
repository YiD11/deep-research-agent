import os

from dotenv import load_dotenv

from ui import create_gradio_ui


def main():
    load_dotenv()
    app = create_gradio_ui()
    env = os.getenv("DEEP_RESEARCH_AGENT_ENV")
    debug = env == "debug"
    ip = os.getenv("GRADIO_SERVER_NAME")
    port = int(os.getenv("GRADIO_SERVER_PORT"))
    app.launch(debug=debug, server_name=ip, server_port=port)


if __name__ == "__main__":
    main()
