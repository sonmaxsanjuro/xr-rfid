import time
import board
import busio
import digitalio
from flask import Flask, Response, request
import adafruit_pn532.spi as PN532_SPI


# --- Flask Web Server Setup ---
app = Flask(__name__)

# --- PN532 RFID Reader Setup ---
# SPI communication with the PN532
spi = busio.SPI(board.SCK, board.MOSI, board.MISO)

# The PN532 requires a Chip Select (CS) pin
cs_pin = digitalio.DigitalInOut(board.D8)  # GPIO8

# Create a PN532 object
pn532 = PN532_SPI.PN532_SPI(spi, cs_pin)

# Check for a connection to the PN532
try:
    pn532.firmware_version
    print("Found PN532 with firmware version: {0}.{1}".format(pn532.firmware_version[0], pn532.firmware_version[1]))
except RuntimeError as e:
    print(f"Error connecting to PN532: {e}")
    exit(1)

# Configure the PN532 for reading
pn532.SAM_configuration()

# --- Web Server Routes ---
@app.route('/', methods=['GET'])
def get_rfid_code():
    """
    Waits for an RFID tag with a timeout and returns its UID as plain text.
    """
    print("Waiting for an RFID tag...")
    # Timeout for the RFID reader (in seconds)
    RFID_TIMEOUT = 0.1 
    
    # Read the UID from the presented tag
    uid = pn532.read_passive_target(timeout=RFID_TIMEOUT)

    if uid is not None:
        # Convert the byte array UID to a hex string
        hex_uid = ''.join([f'{i:02x}' for i in uid])
        print(f"Found tag with UID: {hex_uid}")
        return Response(hex_uid, mimetype='text/plain')
    else:
        print("No RFID tag found within the timeout period.")
        return Response('No RFID tag found within the timeout', mimetype='text/plain', status=408)

# --- Main entry point ---
if __name__ == '__main__':
    # Run the web server on all available network interfaces on port 5000
    app.run(host='0.0.0.0', port=5000)