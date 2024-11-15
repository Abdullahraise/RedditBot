import praw
import pandas as pd
import joblib
import smtplib
import logging
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from apscheduler.schedulers.blocking import BlockingScheduler

# Configurations
REDDIT_CLIENT_ID = 'cfynZYiP0bdEz5_Zw9Q1Ug'
REDDIT_CLIENT_SECRET = 'gVrmX4e8CuHxCmn_ux46Bipnjk74GA'
REDDIT_USER_AGENT = 'JobNotifierBot'
EMAIL_ADDRESS = 'abdullahraise2004@gmail.com'
EMAIL_PASSWORD = 'nusyvaeyxbomdhsb'
RECIPIENT_EMAIL = 'raiseabdullah7@gmail.com'
LOG_FILE = 'notifier_bot.log'
SENT_POSTS_FILE = 'sent_posts.txt'  # File to store sent posts' URLs

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=LOG_FILE)

# Define relevant keywords for graphic design jobs
GRAPHIC_DESIGN_KEYWORDS = ["graphic design", "designer", "illustrator", "branding", "visual design", "art direction", "UI"]

# Load labeled data and train classifier if not already trained
def train_classifier():
    """Trains the job classifier based on labeled data."""
    try:
        data = pd.read_csv('labeled_job_posts.csv')
        X = data['text']
        y = data['label']
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X)
        classifier = MultinomialNB()
        classifier.fit(X_train, y)
        joblib.dump(classifier, 'job_classifier_model.pkl')
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
        logging.info("Model and vectorizer saved successfully!")
        print("Training complete! Model is saved and ready for use.")  # Notify that training is done
    except Exception as e:
        logging.error(f"Error during training: {e}")

# Load classifier and vectorizer
def load_classifier():
    """Loads the trained classifier and vectorizer."""
    try:
        classifier = joblib.load('job_classifier_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return classifier, vectorizer
    except Exception as e:
        logging.error(f"Error loading classifier: {e}")
        raise

def classify_post(text, classifier, vectorizer):
    """Classifies a job post as 'relevant' or 'irrelevant'."""
    post_vectorized = vectorizer.transform([text])
    return classifier.predict(post_vectorized)[0]

def send_email(subject, body, is_initial=False):
    """Sends an email notification with job posts digest."""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL

        # Connect to Gmail's SMTP server and send the email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        # Log and print success message
        logging.info("Email sent successfully!")
        if is_initial:
            print("Initial email sent successfully!")  # Confirmation for initial email
        else:
            print("Email sent successfully after 4 hours!")  # Confirmation after every 4 hours

    except Exception as e:
        logging.error(f"Error sending email: {e}")
        print(f"Error sending email: {e}")  # Display error in CMD

def is_graphic_design_related(post_text):
    """Checks if the post is related to graphic design using relevant keywords."""
    post_text_lower = post_text.lower()
    return any(keyword in post_text_lower for keyword in GRAPHIC_DESIGN_KEYWORDS)

def check_and_collect_posts(reddit, classifier, vectorizer):
    """Fetches new Reddit posts, classifies them, and collects relevant posts."""
    relevant_posts = []

    try:
        subreddit = reddit.subreddit('forhire')
        # Fetch maximum posts available on the subreddit
        for post in subreddit.new(limit=None):  # Set limit to None to fetch as many posts as possible
            if "[Hiring]" in post.title:
                post_text = f"{post.title} {post.selftext}"
                
                # Filter for relevance to graphic design
                if is_graphic_design_related(post_text) and classify_post(post_text, classifier, vectorizer) == 'relevant':
                    relevant_posts.append({'title': post.title, 'url': post.url})

    except Exception as e:
        logging.error(f"Error checking Reddit posts: {e}")

    return relevant_posts

def load_sent_posts():
    """Loads the list of URLs of already sent posts."""
    try:
        with open(SENT_POSTS_FILE, 'r') as file:
            sent_posts = file.read().splitlines()
        return sent_posts
    except FileNotFoundError:
        return []  # Return empty list if file doesn't exist

def save_sent_posts(sent_posts):
    """Saves the URLs of the posts that have been sent."""
    with open(SENT_POSTS_FILE, 'w') as file:
        file.write("\n".join(sent_posts))

def send_digest_email(posts, is_initial=False):
    """Sends a digest email of job posts collected."""
    if not posts:
        logging.info("No new relevant posts to send.")
        return
    
    body = "Here are the latest job posts that match your criteria:\n\n"
    sent_posts = load_sent_posts()  # Load already sent posts to avoid duplicates
    new_posts = []

    for post in posts:
        if post['url'] not in sent_posts:  # Check if the post is already sent
            body += f"Title: {post['title']}\nURL: {post['url']}\n\n"
            new_posts.append(post['url'])

    if new_posts:
        send_email("Job Posts Digest", body, is_initial)
        sent_posts.extend(new_posts)  # Add the new posts to the sent list
        save_sent_posts(sent_posts)  # Save the updated list of sent posts

def scheduled_task():
    """Runs the task every 4 hours."""
    classifier, vectorizer = load_classifier()
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)

    relevant_posts = check_and_collect_posts(reddit, classifier, vectorizer)
    send_digest_email(relevant_posts, is_initial=False)

if __name__ == "__main__":
    # Load the classifier and vectorizer immediately and send the email
    classifier, vectorizer = load_classifier()
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)

    relevant_posts = check_and_collect_posts(reddit, classifier, vectorizer)
    send_digest_email(relevant_posts, is_initial=True)

    # Scheduler to run the task every 4 hours
    scheduler = BlockingScheduler()
    scheduler.add_job(scheduled_task, 'interval', hours=4)  # Runs every 4 hours
    scheduler.start()
