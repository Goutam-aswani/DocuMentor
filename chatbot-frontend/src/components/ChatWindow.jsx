// // src/components/ChatWindow.jsx
// import { Input } from "./ui/input.jsx"
// import { Button } from "./ui/button"
// import { useState, useEffect, useRef } from 'react';
// import { ChevronRightIcon } from "lucide-react";

// import { Send, MessageCircle, Paperclip} from 'lucide-react';
// import ChatMessage from './ChatMessage';

// export default function ChatWindow({ messages, onSendMessage, isLoading, onFileUpload, fileInputRef }) {
//     const [input, setInput] = useState('');
//     const messagesEndRef = useRef(null);

//     useEffect(() => {
//         messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//     }, [messages]);

//     const handleSend = () => {
//         if (input.trim()) {
//             onSendMessage(input);
//             setInput('');
//         }
//     };

//     return (
//         <div className="flex-1 flex flex-col bg-gradient-to-br from-background to-secondary">
//             <div className="flex-1 p-6 overflow-y-auto">
//                 {messages.length === 0 && !isLoading && (
//                     <div className="flex flex-col items-center justify-center h-full text-hsl(var(--muted-foreground))">
//                         <MessageCircle className="w-20 h-20 mb-4 opacity-20" />
//                         <h2 className="text-2xl font-semibold">Welcome to the Chatbot</h2>
//                         <p className="text-hsl(var(--muted-foreground))">Start a new conversation or select one from your history.</p>
//                     </div>
//                 )}
//                 {messages.map((msg) => <ChatMessage key={msg.id || Date.now()} message={msg} />)}
//                 {isLoading && (
//                     <div className="flex items-center gap-2 text-muted-foreground text-sm mt-2">
//                         Bot is typing...
//                     </div>
//                 )}
//                 <div ref={messagesEndRef} />
//             </div>
//             <div className="items-center justify-center ">
//                     {/* <textarea
//                         value={input}
//                         onChange={(e) => setInput(e.target.value)}
//                         onKeyPress={(e) => {
//                             if (e.key === 'Enter' && !e.shiftKey) {
//                                 e.preventDefault();
//                                 handleSend();
//                                 }
//                                 }}
//                                 placeholder="Type your message..."
//                                 className="w-full pl-4 pr-16 py-3 text-black bg-hsl(var(--input)) border border-transparent rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none transition-all"
//                                 rows="1"
//                                 onInput={(e) => {
//                                     e.target.style.height = 'auto';
//                                     e.target.style.height = `${e.target.scrollHeight}px`;
//                                     }}
//                                     /> */}
//                 {/* 3. Add the Hidden File Input */}
//                 <input
//                     type="file"
//                     ref={fileInputRef}
//                     onChange={onFileUpload}
//                     className="hidden"
//                     accept=".pdf,.txt,.docx" // Specify accepted file types
//                 />

//                 <Button 
//                     variant="ghost" 
//                     size="icon" 
//                     onClick={() => fileInputRef.current && fileInputRef.current.click()}
//                     disabled={isLoading}
//                     className="flex justify-center items-center mx-10"
//                 >
//                     <Paperclip className="w-6 h-6 text-muted-foreground" />
//                 </Button>

//                 <div className="flex items-center justify-center mb-5   ">
//                     <Input
//                         type="text"
//                         placeholder="Type your message..."
//                         value={input}
//                         size="sm"
//                         onChange={(e) => setInput(e.target.value)}
//                         onKeyPress={(e) => {
//                             if (e.key === 'Enter' && !e.shiftKey) {
//                                 e.preventDefault();
//                                 handleSend();
//                             }
//                         }}
//                         className="flex-1 px-10 py-6 mx-20 text-left" // Add padding on the right for the button
//                     />
//                     <button className="p-2 text-muted-foreground hover:text-foreground  transition-colors duration-200 ease-in-out">
//                         <Send className="w-6 h-6  rotate-45"
//                         onClick={handleSend}/>
//                     </button>
                       
//                 </div>
//             </div>
//         </div>
//     );
// }


// src/components/ChatWindow.jsx
import { Input } from "./ui/input.jsx"
import { Button } from "./ui/button"
import { useState, useEffect, useRef } from 'react';
import { ChevronRightIcon } from "lucide-react";

import { Send, MessageCircle, Paperclip} from 'lucide-react';
import ChatMessage from './ChatMessage';

export default function ChatWindow({ messages, onSendMessage, isLoading, onFileUpload, fileInputRef, chatInputRef }) {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);
    const localInputRef = useRef(null);

    // Use the passed ref or create a local one
    const inputRef = chatInputRef || localInputRef;

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const handleSend = () => {
        if (input.trim()) {
            onSendMessage(input);
            setInput('');
        }
    };

    return (
        <div className="flex-1 flex flex-col bg-gradient-to-br from-background via-background to-secondary/20 min-h-0">
            {/* Messages Area */}
            <div className="flex-1 p-6 overflow-y-auto min-h-0">
                {messages.length === 0 && !isLoading && (
                    <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                        <div className="relative mb-8">
                            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full blur-xl"></div>
                            <MessageCircle className="relative w-24 h-24 text-muted-foreground/40" />
                        </div>
                        <h2 className="text-3xl font-bold bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text text-transparent mb-2">
                            Welcome to the Chatbot
                        </h2>
                        <p className="text-muted-foreground text-lg text-center max-w-md">
                            Start a new conversation or select one from your history.
                        </p>
                    </div>
                )}
                
                <div className="space-y-4">
                    {messages.map((msg) => (
                        <div key={msg.id || Date.now()} className="fade-in">
                            <ChatMessage message={msg} />
                        </div>
                    ))}
                </div>
                
                {isLoading && (
                    <div className="flex items-center gap-3 text-muted-foreground text-sm mt-6 mb-4">
                        <div className="flex space-x-1">
                            <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                            <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                            <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce"></div>
                        </div>
                        <span>Bot is typing...</span>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>
          <div className="w-full bg-[hsl(var(--background))] border-t border-[hsl(var(--border))]"> {/* Use CSS variables */}
      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={onFileUpload}
        className="hidden"
        accept=".pdf,.txt,.docx"
      />

      <div className="px-4 py-4 sm:px-6">
        <div className="max-w-4xl mx-auto">
          {/* Main Input Container */}
          <div className="flex items-end gap-3">
            {/* File Upload Button */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              // Use CSS variables for file upload button
              className="shrink-0 h-12 w-12 rounded-xl border border-[hsl(var(--border))] hover:border-[hsl(var(--border-hover))] hover:bg-[hsl(var(--secondary))] transition-all duration-200 bg-[hsl(var(--secondary))] shadow-lg"
            >
              {/* Use CSS variables for text color */}
              <Paperclip className="w-5 h-5 text-[hsl(var(--muted-foreground))]" />
            </Button>

            {/* Input Field Container */}
            <div className="relative flex-1">
              <div className="relative rounded-xl border border-[hsl(var(--border))] bg-[hsl(var(--input))] shadow-lg focus-within:border-[hsl(var(--ring))] focus-within:ring-1 focus-within:ring-[hsl(var(--ring))]/50 transition-all duration-200"> {/* Use CSS variables */}
                <textarea
                  ref={inputRef}
                  placeholder="Type your message..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  disabled={isLoading}
                  rows={1}
                  // Use CSS variables for text and placeholder color
                  className="w-full resize-none rounded-xl px-4 py-3 pr-12 text-[hsl(var(--foreground))] placeholder:text-[hsl(var(--muted-foreground))] focus:outline-none bg-transparent min-h-[48px] max-h-32"
                  style={{
                    scrollbarWidth: 'thin',
                    // Use CSS variables for scrollbar color
                    scrollbarColor: 'hsl(var(--muted-foreground)) transparent'
                  }}
                />

                {/* Send Button */}
                <div className="absolute bottom-2 right-2">
                  <button
                    onClick={handleSend}
                    disabled={isLoading || !input.trim()}
                    size="icon"
                    // Use CSS variables for send button colors
                    className="h-10 w-10 rounded-lg bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary))/90] disabled:bg-[hsl(var(--muted))] disabled:cursor-not-allowed transition-all duration-200 shadow-lg flex items-center justify-center"
                  >
                    {isLoading ? (
                      <div className="w-4 h-4 border-2 border-[hsl(var(--primary-foreground))] border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Send className="w-4 h-4 text-[hsl(var(--primary-foreground))]" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Input Hint */}
          <div className="flex justify-center mt-3">
            {/* Use CSS variables for text and background color */}
            <p className="text-xs text-[hsl(var(--muted-foreground))] bg-[hsl(var(--secondary))] px-3 py-1 rounded-full border border-[hsl(var(--border))]">
              Press Enter to send • Shift + Enter for new line
            </p>
          </div>
        </div>
      </div>
    </div>
    </div>
  );
}


// {/* 
//             {/* Input Area */}
//             <div className="border-t border-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
//                 {/* Hidden File Input */}
//                 <input
//                     type="file"
//                     ref={fileInputRef}
//                     onChange={onFileUpload}
//                     className="hidden"
//                     accept=".pdf,.txt,.docx"
//                 />

//                 <div className="p-6">
//                     <div className="flex items-end gap-3 max-w-4xl mx-auto">
//                         {/* File Upload Button */}
//                         <Button 
//                             variant="ghost" 
//                             size="icon" 
//                             onClick={() => fileInputRef.current && fileInputRef.current.click()}
//                             disabled={isLoading}
//                             className="shrink-0 h-12 w-12 rounded-xl border border-border/50 hover:border-border hover:bg-secondary/50 transition-all duration-200"
//                         >
//                             <Paperclip className="w-5 h-5 text-muted-foreground" />
//                         </Button>

//                         {/* Input Field */}
//                         <div className="relative flex-1  rounded-xl border border-border/50 bg-secondary/30 backdrop-blur focus-within:bg-secondary/50 focus-within:border-ring/50 transition-all duration-200">
//                             <input
//                                 type="text"
//                                 placeholder="Type your message..."
//                                 value={input}
//                                 onChange={(e) => setInput(e.target.value)}
//                                 onKeyPress={(e) => {
//                                     if (e.key === 'Enter' && !e.shiftKey) {
//                                         e.preventDefault();
//                                         handleSend();
//                                     }
//                                 }}
//                                 disabled={isLoading}
//                                 className="h-12 pl-4 pr-14 rounded-xl border-border/50 bg-secondary/30 backdrop-blur focus:bg-secondary/50 focus:border-ring/50 transition-all duration-200 text-foreground placeholder:text-muted-foreground/70"
//                             />
                            
//                             {/* Send Button */}
//                             <Button
//                                 onClick={handleSend}
//                                 disabled={isLoading || !input.trim()}
//                                 size="icon"
//                                 className="absolute right-2 top-2 h-8 w-8 rounded-lg bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
//                             >
//                                 <Send className="w-4 h-4" />
//                             </Button>
//                         </div>
//                     </div>

//                     {/* Input Hint */}
//                     <div className="flex justify-center mt-3">
//                         <p className="text-xs text-muted-foreground/60">
//                             Press Enter to send • Shift + Enter for new line
//                         </p>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// } */}