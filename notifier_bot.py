import praw
import pandas as pd
import joblib
import smtplib
import logging
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from time import sleep, time
import os

# Configurations
REDDIT_CLIENT_ID = 'cfynZYiP0bdEz5_Zw9Q1Ug'
REDDIT_CLIENT_SECRET = 'gVrmX4e8CuHxCmn_ux46Bipnjk74GA'
REDDIT_USER_AGENT = 'JobNotifierBot'
EMAIL_ADDRESS = 'abdullahraise2004@gmail.com'
EMAIL_PASSWORD = 'nusyvaeyxbomdhsb'
RECIPIENT_EMAIL = 'raiseabdullah7@gmail.com'
LOG_FILE = 'notifier_bot.log'

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=LOG_FILE)

# Define relevant keywords for graphic design jobs
GRAPHIC_DESIGN_KEYWORDS = ["graphic design", "designer", "illustrator", "branding", "visual design", "art direction", "UI"]

# Load labeled data and train classifier if not already trained
def train_classifier():
    """Trains the job classifier based on labeled data."""
    try:
        data = pd.read_csv('data/labeled_job_posts.csv')
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

# Track sent post IDs to avoid duplicates
def load_sent_post_ids():
    if os.path.exists("sent_post_ids.txt"):
        with open("sent_post_ids.txt", "r") as file:
            return set(file.read().splitlines())
    return set()

def save_sent_post_ids(new_post_ids):
    with open("sent_post_ids.txt", "a") as file:
        for post_id in new_post_ids:
            file.write(f"{post_id}\n")

def send_email(subject, body):
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
        print("Email sent successfully!")  # Display message in CMD

    except Exception as e:
        logging.error(f"Error sending email: {e}")
        print(f"Error sending email: {e}")  # Display error in CMD


def is_graphic_design_related(post_text):
    """Checks if the post is related to graphic design using relevant keywords."""
    post_text_lower = post_text.lower()
    return any(keyword in post_text_lower for keyword in GRAPHIC_DESIGN_KEYWORDS)

def check_and_collect_posts(reddit, classifier, vectorizer, sent_post_ids):
    """Fetches new Reddit posts, classifies them, and collects relevant posts."""
    relevant_posts = []
    new_sent_post_ids = set()

    try:
        subreddit = reddit.subreddit('forhire')
        # Fetch maximum posts available on the subreddit
        for post in subreddit.new(limit=None):  # Set limit to None to fetch as many posts as possible
            if post.id not in sent_post_ids and "[Hiring]" in post.title:
                post_text = f"{post.title} {post.selftext}"
                
                # Filter for relevance to graphic design
                if is_graphic_design_related(post_text) and classify_post(post_text, classifier, vectorizer) == 'relevant':
                    relevant_posts.append({'title': post.title, 'url': post.url})
                    new_sent_post_ids.add(post.id)

    except Exception as e:
        logging.error(f"Error checking Reddit posts: {e}")

    sent_post_ids.update(new_sent_post_ids)
    save_sent_post_ids(new_sent_post_ids)
    
    return relevant_posts

def send_digest_email(posts):
    """Sends a digest email of job posts collected."""
    if not posts:
        logging.info("No new relevant posts to send.")
        return
    
    body = "Here are the latest job posts that match your criteria:\n\n"
    for post in posts:
        body += f"Title: {post['title']}\nURL: {post['url']}\n\n"
    print("Preparing to send email with job post updates...")  # Notify before sending email
    send_email("Job Posts Digest", body)

# Main function to run the bot
def main():
    # Uncomment this line the first time to train the model     train_classifier()

    classifier, vectorizer = load_classifier()
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)

    sent_post_ids = load_sent_post_ids()
    
    # Run the first check and send email immediately
    relevant_posts = check_and_collect_posts(reddit, classifier, vectorizer, sent_post_ids)
    send_digest_email(relevant_posts)
    
    while True:
        start_time = time()
        
        # Run the job check and send emails every 2 minutes
        relevant_posts = check_and_collect_posts(reddit, classifier, vectorizer, sent_post_ids)
        send_digest_email(relevant_posts)
        
        # Calculate elapsed time and wait until 2 minutes have passed
        elapsed = time() - start_time
        sleep(max(0, 4 * 3600 - elapsed))

if __name__ == "__main__":
    main()
