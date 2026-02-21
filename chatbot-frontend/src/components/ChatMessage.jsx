// src/components/ChatMessage.jsx

import { useState } from 'react';
import { Bot, User, Clipboard, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';

// --- Custom Code Block Component ---
const CodeBlock = ({ node, inline, className, children, ...props }) => {
    const [isCopied, setIsCopied] = useState(false);
    const match = /language-(\w+)/.exec(className || '');
    const language = match ? match[1] : 'text';

    const handleCopy = () => {
        const codeString = String(children).replace(/\n$/, '');
        navigator.clipboard.writeText(codeString);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000); // Reset after 2 seconds
    };

    if (inline) {
        // Style for inline code snippets
        return <code className="bg-gray-700 text-sm rounded px-1 py-0.5">{children}</code>;
    }

    return (
        <div className="bg-secondary rounded-lg my-4 overflow-hidden border border-border shadow-md">
            <div className="flex justify-between items-center px-4 py-2 bg-muted">
                <span className="text-xs font-sans text-muted-foreground">{language}</span>
                <button onClick={handleCopy} className="text-muted-foreground hover:text-foreground transition-colors flex items-center">
                    {isCopied ? (
                        <span className="flex items-center text-xs text-green-400">
                            <Check className="w-4 h-4 mr-1" /> Copied!
                        </span>
                    ) : (
                        <span className="flex items-center text-xs">
                            <Clipboard className="w-4 h-4 mr-1" /> Copy
                        </span>
                    )}
                </button>
            </div>
            {/* <pre className="p-4 overflow-x-auto text-sm text-foreground" {...props}> */}
            <pre className="p-4 overflow-x-auto text-sm text-foreground whitespace-pre-wrap" {...props}>

                <code>{children}</code>
            </pre>
        </div>
    );
};


export default function ChatMessage({ message }) {
    const isUser = message.role === 'user';
const [isCopied, setIsCopied] = useState(false);

    const handleCopyMessage = () => {
        navigator.clipboard.writeText(message.content);
        setIsCopied(true);
 setTimeout(() => setIsCopied(false), 2000); // Reset after 2 seconds
    };

    // Format timestamp if it exists
 const formattedTimestamp = message.timestamp ? format(new Date(message.timestamp), 'p') : null; // Using 'p' for short time format (e.m., 11:59 PM)
    

    
    return (
        <div className={`flex items-start gap-4 my-4 fade-in ${isUser ? 'justify-end' : ''}`}>
            {!isUser && 
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0 shadow-lg">
 <Bot className="w-6 h-6 text-white" />
                </div>
            }
            <div className={`max-w-[60%] p-4 rounded-2xl shadow-sm relative ${isUser ? 'bg-gradient-to-r from-primary to-primary/90 text-primary-foreground rounded-br-none' : 'bg-secondary text-secondary-foreground rounded-bl-none'}`}>
                {/* Copy Button for non-code messages */}
                {!isUser && (
                    <button onClick={handleCopyMessage} className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-foreground text-xs">
                        {isCopied ? <Check className="w-3 h-3" /> : <Clipboard className="w-3 h-3" />}
                    </button>
                )}

                {/* Conditional Image Rendering */}
                {message.imageUrl && (
                    <div className="mb-4">
                        <img src={message.imageUrl} alt="message attachment" className="max-w-full h-auto rounded-md" />
                    </div>
                )}
                <ReactMarkdown
                    rehypePlugins={[rehypeHighlight]}
                    components={{
                        pre: CodeBlock,
                        p: ({node, ...props}) => <p className="mb-4 last:mb-0" {...props} />,
                        ul: ({node, ...props}) => <ul className="list-disc list-inside mb-4" {...props} />,
                        ol: ({node, ...props}) => <ol className="list-decimal list-inside mb-4" {...props} />,
                        li: ({node, ...props}) => <li className="mb-1" {...props} />,
                    }}
                >
                    {message.content}
                </ReactMarkdown>
                                {/* Timestamp */}
                {formattedTimestamp && (
                    <div className={`text-[0.65rem] mt-2 ${isUser ? 'text-primary-foreground/70 text-right' : 'text-muted-foreground text-left'}`}>
                        {formattedTimestamp}
                    </div>
                )}
            </div>
            {isUser &&
                <div className="w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0 shadow-lg">
                    <User className="w-5 h-5 text-white"/>
                </div>
            }
        </div>
    );
};
