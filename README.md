# 🕵️‍♂️ Slack Off Meter

> *Because sometimes we all need a digital supervisor with a sense of humor*

A fun productivity tool that uses computer vision to detect when you're not looking at your work and gently (or not so gently) reminds you to get back to business. It's like having your boss watching over your shoulder, but without the awkward small talk and coffee breath.

## 🤔 What is this madness?

Slack Off Meter uses your webcam to:

1. Detect when you're present at your computer
2. Track if you're actually looking at the screen
3. Notice when you pick up your phone
4. Call you out when you've been slacking for too long

Perfect for:
- Remote workers with too much freedom
- Students who should be studying but find themselves watching cat videos
- Developers who spend more time on Reddit than coding
- Anyone who wants to improve their focus while keeping their sense of humor

## 🔧 Technologies

- **Python** - Because we like our snakes digital
- **YOLO v8** - For detecting people and phones (and your lack of productivity)
- **MediaPipe** - For face mesh analysis and judging your life choices
- **OpenCV** - For showing you what you look like when procrastinating
- **PyGame** - For making annoying sounds when you've been slacking off

## 📋 Requirements

- Python 3.8+
- A webcam
- A sense of humor
- A desire to be more productive
- The ability to handle hard truths about your work habits

## 🚀 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/slack-off-meter.git

# Navigate to the directory
cd slack-off-meter

# Install required packages
pip install -r requirements.txt
# Or install individually:
pip install opencv-python mediapipe pygame ultralytics numpy
```

## 📖 Usage

```bash
# Run the application
python slack_off_meter.py
```

When starting, you'll be prompted to enter how long you want the "slack time threshold" to be (in seconds). This is how long you can look away or use your phone before being reminded to get back to work.

### Controls

- **q**: Quit the application (but not your responsibilities)
- **d**: Toggle debug mode to see what's being detected
- **+/-**: Adjust face detection sensitivity

### Features

- **Customizable slack threshold**: Set how long you can slack before alerts trigger
- **Real-time monitoring**: Tracks your focus in real-time
- **Visual feedback**: See exactly what the system is detecting
- **Customizable alerts**: Get humorously shamed when you slack off
- **Session tracking**: Get stats on your productivity session
- **CSV logging**: Track your productivity trends over time

## 🧠 How It Works

1. **Person Detection**: Uses YOLO to detect if you're in front of your computer
2. **Face Orientation**: Uses MediaPipe face mesh to determine if you're looking at the screen
3. **Phone Detection**: Spots when you've picked up your phone
4. **Slacking Timer**: Counts how long you've been unfocused
5. **Alert System**: Notifies you when you've been slacking for too long

## 🏆 Productivity Achievement Levels

Based on your focus percentage:

- **0-20%**: Procrastination Prodigy
- **21-40%**: Distraction Enthusiast
- **41-60%**: Occasional Worker
- **61-80%**: Focus Apprentice
- **81-95%**: Productivity Warrior
- **96-100%**: Either Extremely Focused or You Taped a Photo of Yourself to the Webcam

## ⚠️ Disclaimer

This tool was created for fun and research purposes. Use it to boost your productivity, not to judge yourself too harshly. Remember, even the most productive people take breaks!

The creators of Slack Off Meter are not responsible for:
- Bruised egos
- Existential crises about your work habits
- The realization that you spend 90% of your day looking at social media
- Any productivity-induced euphoria

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more humorous alert messages
- Improve detection algorithms
- Create fancy visualizations for productivity stats
- Add customization options
- Add support for different languages

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

> "I'm not procrastinating, I'm providing my future self with motivation through deadline-induced panic."
>  — Every programmer ever

Made with ❤️ and a desperate need to stay focused
