'use client';

import { useEffect, useRef, useCallback, useState } from 'react';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

interface WebSocketOptions {
  url: string;
  onMessage?: (data: any) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface WebSocketState {
  isConnected: boolean;
  lastMessage: any;
  error: Event | null;
  reconnectCount: number;
}

export function useWebSocket(options: WebSocketOptions) {
  const {
    url,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    reconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 10,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const reconnectCountRef = useRef(0);

  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    lastMessage: null,
    error: null,
    reconnectCount: 0,
  });

  const connect = useCallback(() => {
    if (typeof window === 'undefined') return;

    try {
      const token = localStorage.getItem('auth_token');
      const wsUrl = token ? `${url}?token=${token}` : url;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        reconnectCountRef.current = 0;
        setState((prev) => ({
          ...prev,
          isConnected: true,
          error: null,
          reconnectCount: 0,
        }));
        onConnect?.();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setState((prev) => ({ ...prev, lastMessage: data }));
          onMessage?.(data);
        } catch {
          setState((prev) => ({ ...prev, lastMessage: event.data }));
          onMessage?.(event.data);
        }
      };

      ws.onerror = (error) => {
        setState((prev) => ({ ...prev, error }));
        onError?.(error);
      };

      ws.onclose = () => {
        setState((prev) => ({ ...prev, isConnected: false }));
        onDisconnect?.();

        if (reconnect && reconnectCountRef.current < maxReconnectAttempts) {
          reconnectCountRef.current += 1;
          setState((prev) => ({
            ...prev,
            reconnectCount: reconnectCountRef.current,
          }));
          reconnectTimerRef.current = setTimeout(connect, reconnectInterval);
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('WebSocket connection error:', error);
    }
  }, [url, onMessage, onConnect, onDisconnect, onError, reconnect, reconnectInterval, maxReconnectAttempts]);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof data === 'string' ? data : JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    ...state,
    sendMessage,
    disconnect,
    reconnect: connect,
  };
}

export function usePredictionUpdates(onUpdate?: (prediction: any) => void) {
  return useWebSocket({
    url: `${WS_URL}/ws/predictions`,
    onMessage: onUpdate,
  });
}

export function useOddsUpdates(onUpdate?: (odds: any) => void) {
  return useWebSocket({
    url: `${WS_URL}/ws/odds`,
    onMessage: onUpdate,
  });
}
