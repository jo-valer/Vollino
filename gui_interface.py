import tkinter as tk
import threading
import queue

from tkinter import ttk, scrolledtext
from datetime import datetime

from components.prompts import DEFAULT_SYSTEM_TURNS


class VollinoGUI:
    def __init__(self, callback_func, translator=None, user_settings=None):
        """
        Initialize the Vollino GUI interface.
        
        Args:
            callback_func: Function to call when user sends a message
            translator: Translation function for multilingual support
            user_settings: User settings dictionary
        """
        self.callback_func = callback_func
        self.translator = translator
        self.user_settings = user_settings or {}
        self.message_queue = queue.Queue()
        self.current_checkout_button = None  # Track current checkout button
        self.current_help_text = None  # Track current help text
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Vollino Chat")
        self.root.geometry("800x600")
        self.root.configure(bg="#0e6686")
        
        # Set up the UI
        self.setup_ui()
        
        # Start checking for messages
        self.check_queue()
        
    def setup_ui(self):
        """Set up the user interface elements."""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsiveness
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="Vollino Chat", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT)
        
        # User info
        user_lang = self.user_settings.get("language", "en")
        user_info = ttk.Label(
            header_frame, 
            text=f"Language: {user_lang.upper()}", 
            font=("Arial", 10)
        )
        user_info.pack(side=tk.RIGHT)
        
        # Chat area
        chat_frame = ttk.Frame(main_frame)
        chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat display with scrollbar
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg="white",
            fg="black",
            relief=tk.FLAT,
            borderwidth=1
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chat_display.config(state=tk.DISABLED)
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # User input entry
        self.user_input = ttk.Entry(
            input_frame, 
            font=("Arial", 11),
            width=50
        )
        self.user_input.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message)
        
        # Send button
        self.send_button = ttk.Button(
            input_frame, 
            text="Send", 
            command=self.send_message
        )
        self.send_button.grid(row=0, column=1)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(
            status_frame, 
            text="Ready", 
            font=("Arial", 9),
            foreground="gray"
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Clear button
        clear_button = ttk.Button(
            status_frame, 
            text="Clear Chat", 
            command=self.clear_chat
        )
        clear_button.pack(side=tk.RIGHT)
        
        # Set focus to input field
        self.user_input.focus()
        
    def add_message(self, sender, message, color=None, show_checkout=False):
        """Add a message to the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Configure tags for styling
        if sender == "Vollino":
            tag = "bot"
            if not color:
                color = "#0066cc"
        else:
            tag = "user"
            if not color:
                color = "#009900"
        
        self.chat_display.tag_configure(tag, foreground=color, font=("Arial", 11, "bold"))
        self.chat_display.tag_configure("timestamp", foreground="gray", font=("Arial", 9))
        self.chat_display.tag_configure("message", foreground="black", font=("Arial", 11))
        
        # Insert the message
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.chat_display.insert(tk.END, f"{sender}: ", tag)
        self.chat_display.insert(tk.END, f"{message}", "message")
        
        # Add checkout button inline if needed
        if show_checkout and sender == "Vollino":
            self.chat_display.insert(tk.END, "\n")
            
            # Create a frame for the button and text
            button_frame = tk.Frame(self.chat_display, bg="white")
            
            # Create the checkout button
            checkout_button = tk.Button(
                button_frame,
                text="🛒 Checkout",
                command=lambda: self.handle_checkout(checkout_button),
                bg="#0066cc",
                fg="white",
                font=("Arial", 10, "bold"),
                relief=tk.RAISED,
                borderwidth=2,
                padx=10,
                pady=5
            )
            checkout_button.pack(side=tk.LEFT)
            
            # Store reference to current checkout button
            self.current_checkout_button = checkout_button
            
            # Add helpful text next to the button
            help_text = tk.Label(
                button_frame,
                text="Not quite right? Just keep chatting to make changes.",
                bg="white",
                fg="#666666",
                font=("Arial", 9, "italic"),
                padx=10
            )
            help_text.pack(side=tk.LEFT, anchor=tk.CENTER)
            
            # Store reference to help text so we can hide it later
            self.current_help_text = help_text
            
            # Insert the button widget into the text
            self.chat_display.window_create(tk.END, window=button_frame)
        
        self.chat_display.insert(tk.END, "\n\n")
        
        # Scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def send_message(self, event=None):
        """Handle sending a user message."""
        message = self.user_input.get().strip()
        if not message:
            return
            
        # Hide current checkout button if it exists
        self.hide_current_checkout_button()
            
        # Clear input field
        self.user_input.delete(0, tk.END)
        
        # Disable send button temporarily
        self.send_button.config(state="disabled")
        self.status_label.config(text="Processing...")
        
        # Add user message to display
        display_message = message
        if self.user_settings.get("language") == "it":
            display_message = message  # Show original Italian input
            
        self.add_message("You", display_message, "#009900")
        
        # Process message in separate thread to avoid blocking UI
        threading.Thread(
            target=self.process_message, 
            args=(message,), 
            daemon=True
        ).start()
        
    def handle_checkout(self, button):
        """Handle checkout button click."""
        # Hide the help text
        if self.current_help_text and self.current_help_text.winfo_exists():
            try:
                self.current_help_text.destroy()
            except tk.TclError:
                pass  # Text might have been destroyed already
            self.current_help_text = None
        
        # Change button appearance to green and disable it
        button.config(
            bg="#28a745",  # Green color
            text="✓ Checkout!",
            state="disabled",
            relief=tk.FLAT
        )
        
        # Clear the current checkout button reference since it's now disabled
        self.current_checkout_button = None
        
        # Add system response for checkout
        checkout_msg = DEFAULT_SYSTEM_TURNS["RESTART"]
        if self.user_settings.get("language") == "it" and self.translator:
            try:
                checkout_msg = self.translator(checkout_msg, target_lang="it")
            except:
                pass  # Fall back to English if translation fails
        
        self.add_message("Vollino", checkout_msg, "#0066cc")
        
    def hide_current_checkout_button(self):
        """Hide the current checkout button if it exists and is still clickable."""
        if self.current_checkout_button and self.current_checkout_button.winfo_exists():
            try:
                # Only hide if the button is still active (not disabled)
                if str(self.current_checkout_button['state']) != 'disabled':
                    # Get the parent frame and hide it
                    parent_frame = self.current_checkout_button.master
                    if parent_frame:
                        parent_frame.destroy()
                self.current_checkout_button = None
            except tk.TclError:
                # Button might have been destroyed already
                self.current_checkout_button = None
        
    def process_message(self, message):
        """Process the message using the callback function."""
        try:
            # Call the pipeline callback function
            result = self.callback_func(message)
            
            # Queue the response for the main thread
            if isinstance(result, dict):
                self.message_queue.put(("response", result))
            else:
                # Backward compatibility for non-GUI mode
                self.message_queue.put(("response", {"response": result, "show_checkout": False}))
            
        except Exception as e:
            # Queue error message
            self.message_queue.put(("error", f"Error: {str(e)}"))
            
    def add_system_response(self, result):
        """Add system response to the chat."""
        # Extract response and checkout info
        if isinstance(result, dict):
            response = result.get("response", "")
            show_checkout = result.get("show_checkout", False)
        else:
            response = result
            show_checkout = False
            
        # Translate response if needed
        display_response = response
        if self.user_settings.get("language") == "it" and self.translator:
            try:
                display_response = self.translator(response, target_lang="it")
            except:
                pass  # Fall back to English if translation fails
                
        self.add_message("Vollino", display_response, "#0066cc", show_checkout=show_checkout)
        
        # Re-enable send button
        self.send_button.config(state="normal")
        self.status_label.config(text="Ready")
        
    def check_queue(self):
        """Check for messages from background threads."""
        try:
            while True:
                msg_type, content = self.message_queue.get_nowait()
                
                if msg_type == "response":
                    self.add_system_response(content)
                elif msg_type == "error":
                    self.add_message("System", content, "#cc0000")
                    self.send_button.config(state="normal")
                    self.status_label.config(text="Ready")
                    
        except queue.Empty:
            pass
            
        # Schedule next check
        self.root.after(100, self.check_queue)
        
    def clear_chat(self):
        """Clear the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        # Clear checkout button reference when clearing chat
        self.current_checkout_button = None
        self.current_help_text = None
        
    def add_initial_message(self, message):
        """Add an initial system message."""
        if isinstance(message, dict):
            self.add_system_response(message)
        else:
            self.add_system_response({"response": message, "show_checkout": False})
        
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()
        
    def close(self):
        """Close the GUI."""
        self.root.quit()
        self.root.destroy()
