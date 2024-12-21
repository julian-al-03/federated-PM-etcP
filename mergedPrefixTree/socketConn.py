from queue import Queue, Empty
import socket
import struct
import threading
from typing import Dict, List, Tuple
import pickle
import uuid
from customTypes.types import Message, OrgPort, OrgId
import select

class OrgSocket:
    def __init__(self, orgPort: int, orgIdentifier: OrgId, coordinatorPort: int, orgPorts: List[OrgPort], handler) -> None:
        self.orgIdentifier: OrgId = orgIdentifier
        self.orgPort: int = orgPort
        self.coordinatorPort: int = coordinatorPort
        self.orgPorts: List[OrgPort] = orgPorts  # List of (name, port) tuples
        self.connectionMap: Dict[OrgId, socket.socket] = {}  # Map organization names to their sockets
        self.serverSocket: socket.socket = None
        self.stop_event = threading.Event()
        self.listener_thread: threading.Thread = None
        self.peer_connection_threads: List[threading.Thread] = []
        self.handler = handler
        self.responseQueues: Dict[str, Queue] = {}  # Map message ID to a Queue for waiting for responses


    def startServer(self):
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
        self.serverSocket.bind(('127.0.0.1', self.orgPort))
        self.serverSocket.listen()

        try:
            while not self.stop_event.is_set():
                try:
                    conn, addr = self.serverSocket.accept()
                    thread = threading.Thread(target=self.handle_peer_connection, args=(conn, addr))
                    self.peer_connection_threads.append(thread)
                    thread.start()
                except ConnectionAbortedError:
                    # This is expected when the socket is closed
                    break
                except OSError as e:
                    # Handle other potential socket-related errors
                    print(f"Socket error: {e}")
                    break
        finally:
            self.serverSocket.close()

    def handle_peer_connection(self, conn, addr):
        try:
            while not self.stop_event.is_set():
                # Use a non-blocking approach with a timeout to check for shutdown signals
                ready_to_read, _, _ = select.select([conn], [], [], 1)
                if ready_to_read:
                    data = self.recv_data(conn)
                    if data is None:  # Connection closed
                        break
                    message: Message = pickle.loads(data)
                    sender = message.sender

                    messageId = message.id
                    if messageId and messageId in self.responseQueues:
                        self.responseQueues[messageId].put(message)
                    else:
                        # Handle messages asynchronously
                        listener_thread = threading.Thread(target=self.handler, args=(sender, message))
                        listener_thread.start()
        except Exception as e:
            pass
        finally:
            conn.close()


    def recv_data(self, sock):
        raw_data_length = self.recv_all(sock, 4)
        if not raw_data_length:
            return None
        data_length = struct.unpack('!I', raw_data_length)[0]
        serialized_data = self.recv_all(sock, data_length)
        if not serialized_data:
            return None
        return pickle.loads(serialized_data)

    def recv_all(self, sock, length):
        data = b''
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def connect_to_peer(self, orgId: OrgId):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
        peer_ip = '127.0.0.1'
        peer_port = None
        if orgId == 'coordinator':
            peer_port = self.coordinatorPort
        else:
            for org in self.orgPorts:
                if org.orgId == orgId:
                    peer_port = org.port
                    break
        # print(f"Connecting to {orgId} on port {peer_port}")
        sock.connect((peer_ip, peer_port))
        self.connectionMap[orgId] = sock
        # sock.sendall(message.encode())
        # response = sock.recv(1024)
        # print(f"Received from {peer_ip}: {response.decode()}")
        # sock.close()

    def listen(self) -> None:
        while not self.stop_event.is_set():
            data, addr = self.socket.recvfrom(4096**4)  # Buffer size is 4096 bytes
            message: Message = pickle.loads(data)

            messageId = message.id
            if messageId and messageId in self.responseQueues:
                # If it's a reply for a specific message ID, put it in the corresponding queue
                self.responseQueues[messageId].put(message)
            else:
                self.listener_thread = threading.Thread(target=self.handler, args=(message, message.sender))
                self.listener_thread.start()

    def sendToCoordinator(self, message: Message) -> None:
        self.sendToOrg("coordinator", message)

    def sendToOrg(self, orgId: str, message: Message) -> None:
        message.sender = self.orgIdentifier
        # check if connection exists
        if orgId not in self.connectionMap:
            self.connect_to_peer(orgId)
        socket = self.connectionMap[orgId]
        # data = pickle.dumps(message)
        # serialized_data = pickle.dumps(data)
        # data_length = struct.pack('!I', len(serialized_data))
        # socket.sendall(data_length + serialized_data)
        self.send_with_wait(socket, message)

    def send_with_wait(self, socket, message):
        # Serialize the message using pickle
        data = pickle.dumps(message)
        serialized_data = pickle.dumps(data)

        # Prepend the length of the serialized data
        data_length = struct.pack('!I', len(serialized_data))

        # Combine the length header and the serialized data into a single message
        total_data = data_length + serialized_data

        # Loop to ensure all data is sent
        while total_data:
            # Check if the socket is ready for writing (with a 1-second timeout)
            _, writable, _ = select.select([], [socket], [], 1.0)

            if writable:
                try:
                    # Send the data that remains to be sent
                    sent_bytes = socket.send(total_data)
                    total_data = total_data[sent_bytes:]  # Remove sent part
                except socket.error as e:
                    print(f"Socket error: {e}")
                    # Optionally handle specific socket errors like retrying, etc.
                    break
            else:
                # Optional: Add a small delay if you want to avoid tight looping
                # time.sleep(0.01)  # Not strictly necessary, but can help reduce CPU usage
                continue

        # Optionally, you can return True to indicate success or False on failure
        return len(total_data) == 0


    def sendToOrgWaitForResponse(self, orgName: str, message: Message) -> Message:
        """
        Send a message and wait for a response with the same ID.
        Timeout after `timeout` seconds.
        """
        timeout = 300000
        messageId = str(uuid.uuid4())  # Unique ID for this message
        message.id = messageId  # Attach the unique ID to the message

        # Create a queue to wait for the response
        self.responseQueues[messageId] = Queue()

        # Send the message
        self.sendToOrg(orgName, message)

        try:
            # Wait for a response for up to `timeout` seconds
            return self.responseQueues[messageId].get(timeout=timeout)
        except Empty:
            print(f"Timed out waiting for response to message ID {messageId}")
            return None
        finally:
            # Cleanup the queue once we're done waiting
            del self.responseQueues[messageId]

    def start(self) -> None:
        self.listener_thread = threading.Thread(target=self.startServer)
        self.listener_thread.start()
        print(f"Listening on 127.0.0.1:{self.orgPort}")

    def close(self) -> None:
        # Signal threads to stop
        self.stop_event.set()
        # Close the server socket to unblock accept()
        if self.serverSocket:
            self.serverSocket.close()

        # Join listener thread
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join()

        # Join all peer connection threads
        for thread in self.peer_connection_threads:
            if thread.is_alive():
                thread.join()
                
        # Close all connected sockets
        for conn in self.connectionMap.values():
            conn.close()

