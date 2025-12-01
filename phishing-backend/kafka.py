import pandas as pd
import time
import random
import os
import sys

# Configuration
TOPIC_NAME = "phishing-predictions-stream"
BROKER = "kafka-broker:9092"
GROUP_ID = "drift-monitor-group"
LOG_FILE = 'prediction_log.csv'

# ANSI Colors for "Hacker" aesthetic
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def simulate_kafka_stream():
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{Colors.HEADER}{Colors.BOLD}============================================================{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}   APACHE KAFKA STREAMING (v3.2.0)   {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}============================================================{Colors.ENDC}")
    print(f"[*] Connecting to Bootstrap Server: {Colors.CYAN}{BROKER}{Colors.ENDC}...")
    time.sleep(1.5)
    print(f"[*] Joining Consumer Group: {Colors.CYAN}{GROUP_ID}{Colors.ENDC}...")
    time.sleep(1.0)
    print(f"[*] Subscribed to Topic: {Colors.CYAN}{TOPIC_NAME}{Colors.ENDC}")
    print(f"[*] Status: {Colors.GREEN}CONNECTED{Colors.ENDC}")
    print("-" * 60)
    
    if not os.path.exists(LOG_FILE):
        print(f"{Colors.FAIL}No data found in {LOG_FILE}. Generate some traffic first!{Colors.ENDC}")
        return

    try:
        df = pd.read_csv(LOG_FILE)
    except:
        return

    # Calculate Drift Stats accumulator
    total_conf = 0
    count = 0
    
    print(f"{Colors.BOLD}waiting for messages...{Colors.ENDC}\n")
    
    # Simulate streaming rows one by one
    for index, row in df.iterrows():
        # Random delay to simulate network latency/real-time traffic
        delay = random.uniform(0.3, 1.2) 
        time.sleep(delay)
        
        prediction = row.get('prediction', 'Unknown')
        confidence = float(row.get('confidence', 0))
        input_type = row.get('input_type', 'Unknown')
        
        # Accumulate stats
        total_conf += confidence
        count += 1
        avg_conf = total_conf / count
        
        # Construct a fake message offset
        offset = 10420 + index
        partition = index % 3
        
        # Print the "Received" log
        timestamp = time.strftime("%H:%M:%S")
        
        msg_color = Colors.GREEN if prediction == 'Safe' else Colors.FAIL
        
        print(f"[{timestamp}] {Colors.BLUE}[CONSUMER]{Colors.ENDC} Partition:{partition} | Offset:{offset} | Payload: {{ Type: {input_type}, Pred: {msg_color}{prediction}{Colors.ENDC}, Conf: {confidence:.4f} }}")
        
        # Check for drift "Real-time"
        if avg_conf < 0.70 and count > 5:
             print(f"    >>> {Colors.WARNING}[DRIFT ALERT]{Colors.ENDC} Running Avg Confidence dropped to {avg_conf:.4f}")

    print("\n" + "-" * 60)
    print(f"{Colors.GREEN}Stream processing complete. Waiting for new messages...{Colors.ENDC}")
    
    # Simulate waiting
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Disconnecting from Kafka Broker...")
        sys.exit(0)

if __name__ == "__main__":
    simulate_kafka_stream()