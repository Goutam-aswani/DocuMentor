import { useState, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import { api } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { MessageSquare, Zap, Search, Calendar, TrendingUp, Bot } from 'lucide-react';

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0'];

export default function UsageDashboard() {
    const { token } = useAuth();
    const [totals, setTotals] = useState(null);
    const [dailyStats, setDailyStats] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [selectedDays, setSelectedDays] = useState(30);

    useEffect(() => {
        fetchUsageData();
    }, [selectedDays]);

    const fetchUsageData = async () => {
        if (!token) return;
        
        setIsLoading(true);
        try {
            const [totalsResponse, statsResponse] = await Promise.all([
                api.getUsageTotals(token),
                api.getUsageStats(token, selectedDays)
            ]);
            
            setTotals(totalsResponse);
            setDailyStats(statsResponse.stats || []);
        } catch (error) {
            console.error('Failed to fetch usage data:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const formatModelData = (modelUsage) => {
        return Object.entries(modelUsage).map(([model, count]) => ({
            name: model,
            value: count
        }));
    };

    const formatChartData = (stats) => {
        return stats.map(stat => ({
            date: new Date(stat.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            messages: stat.messages_sent,
            tokens: stat.tokens_used,
            searches: stat.web_searches_made,
            sessions: stat.sessions_created
        }));
    };

    if (isLoading) {
        return (
            <div className="min-h-screen bg-[hsl(var(--background))] flex items-center justify-center">
                <div className="text-center">
                    <div className="w-16 h-16 border-4 border-[hsl(var(--primary))] border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-[hsl(var(--muted-foreground))]">Loading usage statistics...</p>
                </div>
            </div>
        );
    }

    const chartData = formatChartData(dailyStats);
    const modelData = totals?.model_usage ? formatModelData(totals.model_usage) : [];

    return (
        <div className="min-h-screen bg-[hsl(var(--background))] p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-[hsl(var(--foreground))] mb-2">Usage Dashboard</h1>
                    <p className="text-[hsl(var(--muted-foreground))]">Track your chatbot usage and analytics</p>
                </div>

                {/* Time Period Selector */}
                <div className="mb-6">
                    <div className="flex gap-2">
                        {[7, 14, 30, 60].map(days => (
                            <button
                                key={days}
                                onClick={() => setSelectedDays(days)}
                                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                                    selectedDays === days
                                        ? 'bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))]'
                                        : 'bg-[hsl(var(--secondary))] text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))]'
                                }`}
                            >
                                {days} days
                            </button>
                        ))}
                    </div>
                </div>

                {/* Overview Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-lg p-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-medium text-[hsl(var(--muted-foreground))] mb-1">Total Messages</p>
                                <p className="text-2xl font-bold text-[hsl(var(--foreground))]">{totals?.total_messages || 0}</p>
                            </div>
                            <MessageSquare className="w-8 h-8 text-blue-500" />
                        </div>
                    </div>

                    <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-lg p-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-medium text-[hsl(var(--muted-foreground))] mb-1">Tokens Used</p>
                                <p className="text-2xl font-bold text-[hsl(var(--foreground))]">{totals?.total_tokens?.toLocaleString() || 0}</p>
                            </div>
                            <Zap className="w-8 h-8 text-yellow-500" />
                        </div>
                    </div>

                    <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-lg p-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-medium text-[hsl(var(--muted-foreground))] mb-1">Sessions Created</p>
                                <p className="text-2xl font-bold text-[hsl(var(--foreground))]">{totals?.total_sessions || 0}</p>
                            </div>
                            <Calendar className="w-8 h-8 text-green-500" />
                        </div>
                    </div>

                    <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-lg p-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-medium text-[hsl(var(--muted-foreground))] mb-1">Web Searches</p>
                                <p className="text-2xl font-bold text-[hsl(var(--foreground))]">{totals?.total_searches || 0}</p>
                            </div>
                            <Search className="w-8 h-8 text-purple-500" />
                        </div>
                    </div>
                </div>

                {/* Charts Row */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                    {/* Daily Activity Chart */}
                    <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-lg p-6">
                        <h3 className="text-lg font-semibold text-[hsl(var(--foreground))] mb-4 flex items-center gap-2">
                            <TrendingUp className="w-5 h-5" />
                            Daily Activity
                        </h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                                <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" />
                                <YAxis stroke="hsl(var(--muted-foreground))" />
                                <Tooltip 
                                    contentStyle={{
                                        backgroundColor: 'hsl(var(--popover))',
                                        border: '1px solid hsl(var(--border))',
                                        borderRadius: '8px'
                                    }}
                                />
                                <Line type="monotone" dataKey="messages" stroke="#8884d8" strokeWidth={2} />
                                <Line type="monotone" dataKey="searches" stroke="#82ca9d" strokeWidth={2} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Model Usage Chart */}
                    <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-lg p-6">
                        <h3 className="text-lg font-semibold text-[hsl(var(--foreground))] mb-4 flex items-center gap-2">
                            <Bot className="w-5 h-5" />
                            Model Usage
                        </h3>
                        {modelData.length > 0 ? (
                            <ResponsiveContainer width="100%" height={300}>
                                <PieChart>
                                    <Pie
                                        data={modelData}
                                        cx="50%"
                                        cy="50%"
                                        labelLine={false}
                                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                        outerRadius={80}
                                        fill="#8884d8"
                                        dataKey="value"
                                    >
                                        {modelData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                        ))}
                                    </Pie>
                                    <Tooltip />
                                </PieChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="flex items-center justify-center h-[300px] text-[hsl(var(--muted-foreground))]">
                                No model usage data available
                            </div>
                        )}
                    </div>
                </div>

                {/* Token Usage Chart */}
                <div className="bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-[hsl(var(--foreground))] mb-4">Token Usage Over Time</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                            <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" />
                            <YAxis stroke="hsl(var(--muted-foreground))" />
                            <Tooltip 
                                contentStyle={{
                                    backgroundColor: 'hsl(var(--popover))',
                                    border: '1px solid hsl(var(--border))',
                                    borderRadius: '8px'
                                }}
                            />
                            <Bar dataKey="tokens" fill="#8884d8" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}
