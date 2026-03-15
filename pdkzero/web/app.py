from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs

from pdkzero.web.session import WebGameSession

INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PDKZero Web PvE</title>
  <style>
    :root { font-family: "Segoe UI", "PingFang SC", sans-serif; color: #1c1f21; background: #ebe6de; }
    body { margin: 0; padding: 8px; min-height: 100vh; }
    .app { max-width: 1260px; margin: 0 auto; display: grid; gap: 10px; }
    .toolbar { display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap; }
    .table-layout { width: 100%; }
    .table-shell { display: grid; grid-template-columns: 290px 1fr 290px; grid-template-rows: 180px 350px minmax(280px, auto); gap: 10px; }
    .seat, .center-board, .player-panel { background: #dcd3ec; border: 1px solid rgba(72, 50, 101, 0.12); border-radius: 10px; box-sizing: border-box; }
    .seat { padding: 10px; overflow: hidden; }
    .seat.top { grid-column: 2; grid-row: 1; }
    .seat.left { grid-column: 1; grid-row: 2; }
    .seat.right { grid-column: 3; grid-row: 2; }
    .center-board { grid-column: 2; grid-row: 2; background: #2b0f59; color: #ffffff; padding: 12px; display: grid; align-content: space-between; }
    .player-panel { grid-column: 2; grid-row: 3; padding: 12px; display: grid; gap: 10px; overflow: visible; }
    .seat-line { font-weight: 700; font-size: 15px; color: #2d2038; margin-bottom: 8px; }
    .seat-line span { color: #2d2038; }
    .board-head { display: grid; gap: 8px; font-size: 14px; }
    .board-label { color: rgba(255,255,255,0.88); }
    .board-selected { display: grid; gap: 4px; align-content: start; }
    .board-selected-label { color: rgba(255,255,255,0.88); font-size: 14px; }
    .selected-summary { color: rgba(255,255,255,0.74); font-size: 13px; }
    .board-center-move { display: grid; justify-items: center; gap: 8px; min-height: 160px; align-content: center; }
    .controls { display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap; }
    .hand-wrap { overflow-x: auto; overflow-y: visible; padding-top: 4px; padding-bottom: 12px; }
    .card-group { display: flex; align-items: flex-end; gap: 8px; min-height: 96px; }
    .card-group.overlap-hand { gap: 0; padding-left: 32px; min-height: 140px; }
    .card-group.overlap-hand .playing-card { margin-left: -32px; }
    .card-group.overlap-hand .playing-card:first-child { margin-left: 0; }
    .card-group.seat-top-group { gap: 0; padding-left: 28px; justify-content: center; min-height: 96px; }
    .card-group.seat-top-group .playing-card { margin-left: -22px; }
    .card-group.seat-top-group .playing-card:first-child { margin-left: 0; }
    .seat-side-grid { display: flex; flex-wrap: wrap; gap: 6px; align-content: flex-start; }
    .card-group.compact { gap: 6px; min-height: 70px; }
    .playing-card { width: 76px; height: 108px; filter: drop-shadow(0 6px 10px rgba(0,0,0,0.16)); cursor: default; transition: transform 0.12s ease, filter 0.12s ease; }
    .playing-card.hand { width: 84px; height: 120px; cursor: pointer; }
    .playing-card.hand:hover { transform: translateY(-6px); }
    .playing-card.hand.selected { transform: translateY(-18px); filter: drop-shadow(0 10px 14px rgba(0,0,0,0.24)); }
    .playing-card.board { width: 72px; height: 102px; }
    .playing-card.selected-board { width: 60px; height: 86px; }
    .playing-card.action { width: 44px; height: 64px; }
    .playing-card.history { width: 40px; height: 58px; }
    .playing-card.seat-back { width: 44px; height: 62px; }
    .playing-card.seat-open { width: 44px; height: 62px; }
    .playing-card.seat-side-open { width: 50px; height: 72px; }
    .playing-card-svg { width: 100%; height: 100%; display: block; }
    .playing-card.red { color: #d11f3a; }
    .playing-card.black { color: #202733; }
    .empty-move { display: inline-flex; align-items: center; justify-content: center; min-width: 72px; min-height: 34px; padding: 0 12px; border-radius: 999px; background: rgba(255,255,255,0.14); color: inherit; font-weight: 700; }
    .seat-stack { display: flex; align-items: flex-end; gap: 0; }
    .seat-stack .playing-card { margin-left: -14px; }
    .seat-stack .playing-card:first-child { margin-left: 0; }
    .flight-canvas { position: fixed; inset: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 20; }
    .seat-pass-overlay { position: fixed; z-index: 24; pointer-events: none; min-width: 74px; padding: 8px 14px; border-radius: 999px; background: rgba(43, 15, 89, 0.92); color: #ffffff; font-weight: 800; letter-spacing: 0.08em; text-align: center; box-shadow: 0 10px 18px rgba(20, 11, 40, 0.28); transform: translate(-50%, -50%) scale(0.92); opacity: 0; transition: opacity 0.12s ease, transform 0.12s ease; }
    .seat-pass-overlay.visible { opacity: 1; transform: translate(-50%, -50%) scale(1); }
    button { border: 1px solid rgba(61, 41, 90, 0.2); border-radius: 8px; padding: 6px 14px; cursor: pointer; background: #f3f0f9; color: #312346; font-size: 14px; }
    button.primary { background: #efe8fb; }
    button:disabled { opacity: 0.48; cursor: not-allowed; }
    .muted { color: rgba(255,255,255,0.74); }
    .status-pill { display: inline-flex; align-items: center; gap: 6px; color: rgba(255,255,255,0.92); }
    @media (max-width: 1100px) {
      .table-shell { grid-template-columns: 1fr; grid-template-rows: auto; }
      .seat.top, .seat.left, .seat.right, .center-board, .player-panel { grid-column: auto; grid-row: auto; }
      .seat-side-grid { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <button id="toggle-hand-visibility">显示四家明牌</button>
      <button id="new-game">新开一局</button>
    </div>
    <div class="table-layout">
      <div class="table-shell">
      <div class="seat top" id="seat-top"></div>
      <div class="seat left" id="seat-left"></div>
      <div class="seat right" id="seat-right"></div>
      <div class="center-board">
        <div class="board-head">
          <div class="status-pill" id="status">加载中...</div>
          <div class="board-selected">
            <div class="board-selected-label">已选:</div>
            <div class="selected-summary" id="selected-summary">-</div>
          </div>
        </div>
        <div class="board-center-move">
          <div id="lead"></div>
          <div id="last-move"></div>
        </div>
      </div>
      <div class="player-panel">
        <div class="seat-line" id="seat-bottom-title"></div>
        <div class="hand-wrap">
          <div class="hand-cards" id="hand-cards"></div>
        </div>
        <div class="controls">
          <button id="hint-action">提示</button>
          <button class="primary" id="play-selected">出牌</button>
          <button id="pass-action">过牌</button>
          <button id="clear-selection">清空</button>
        </div>
      </div>
      </div>
    </div>
    <canvas class="flight-canvas" id="flight-canvas" aria-hidden="true"></canvas>
    <div class="seat-pass-overlay" id="seat-pass-overlay" data-player=""></div>
  </div>
  <script>
    const SUIT_META = {
      C: {symbol: "♣", color: "black"},
      D: {symbol: "♦", color: "red"},
      H: {symbol: "♥", color: "red"},
      S: {symbol: "♠", color: "black"},
    };
    const AI_CHAIN_PAUSE_MS = 1000;
    const AI_MOVE_FLY_MS = 600;
    let revealHands = false;
    let selectedCards = [];
    let latestState = null;
    let isAnimating = false;
    window.__pdkAnimating = false;

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        ...options,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "request failed");
      }
      return data;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function parseCard(card) {
      const suit = card[0];
      const rank = card.slice(1);
      const meta = SUIT_META[suit] || {symbol: "?", color: "black"};
      return {rank, suit, symbol: meta.symbol, color: meta.color};
    }

    function renderPlayingCard(card, variant = "board") {
      const parsed = parseCard(card);
      return `
        <div class="playing-card ${variant} ${parsed.color}" title="${escapeHtml(card)}">
          <svg class="playing-card-svg" viewBox="0 0 120 170" role="img" aria-label="${escapeHtml(card)}">
            <rect x="4" y="4" width="112" height="162" rx="14" fill="#ffffff" stroke="#1f2933" stroke-width="2"/>
            <text x="18" y="28" font-size="22" font-weight="700" fill="currentColor">${escapeHtml(parsed.rank)}</text>
            <text x="18" y="52" font-size="20" fill="currentColor">${parsed.symbol}</text>
            <text x="60" y="94" text-anchor="middle" dominant-baseline="middle" font-size="44" fill="currentColor">${parsed.symbol}</text>
            <g transform="translate(120 170) rotate(180)">
              <text x="18" y="28" font-size="22" font-weight="700" fill="currentColor">${escapeHtml(parsed.rank)}</text>
              <text x="18" y="52" font-size="20" fill="currentColor">${parsed.symbol}</text>
            </g>
          </svg>
        </div>
      `;
    }

    function renderCardBack(variant = "seat-back") {
      return `
        <div class="playing-card ${variant}" title="牌背">
          <svg class="playing-card-svg" viewBox="0 0 120 170" role="img" aria-label="牌背">
            <rect x="4" y="4" width="112" height="162" rx="14" fill="#ffffff" stroke="#1f2933" stroke-width="2"/>
            <rect x="12" y="12" width="96" height="146" rx="10" fill="#173b63"/>
            <rect x="20" y="20" width="80" height="130" rx="8" fill="#244f82"/>
            <path d="M20 40 L100 40 M20 70 L100 70 M20 100 L100 100 M20 130 L100 130" stroke="#9ac3f0" stroke-width="4" opacity="0.6"/>
            <circle cx="60" cy="85" r="18" fill="#d8e6f5" opacity="0.85"/>
          </svg>
        </div>
      `;
    }

    function renderCardGroup(cards, variant = "board", overlap = false, emptyText = "PASS") {
      if (!cards || cards.length === 0) {
        return `<div class="empty-move">${escapeHtml(emptyText)}</div>`;
      }
      const overlapClass = overlap ? " overlap-hand" : "";
      const compactClass = variant === "action" || variant === "history" || variant === "seat-back" ? " compact" : "";
      return `<div class="card-group${overlapClass}${compactClass}">${cards.map(card => renderPlayingCard(card, variant)).join("")}</div>`;
    }

    function renderSeatStack(count) {
      const shown = Math.min(count, 5);
      return `<div class="seat-stack">${Array.from({length: shown}, () => renderCardBack("seat-back")).join("")}</div>`;
    }

    function renderMovePanel(title, move, variant = "board") {
      if (!move) {
        return `<div class="board-label">${escapeHtml(title)}</div><div class="muted">无</div>`;
      }
      return `
        <div class="board-label">${escapeHtml(title)}: ${escapeHtml(move.move_type_name)}</div>
        ${renderCardGroup(move.cards, variant, false, move.text)}
      `;
    }

    function renderMoveSummary(title, entry) {
      if (!entry) {
        return `<div class="muted">${escapeHtml(title)}: 无</div>`;
      }
      return `<div class="muted">${escapeHtml(title)}: p${entry.player + 1} · ${escapeHtml(entry.move.move_type_name)} · ${escapeHtml(entry.move.text)}</div>`;
    }

    function renderSeatCards(seat, state, reveal) {
      const cards = state.all_hands[seat] || [];
      if (!reveal && seat !== 0) {
        if (seat === 2) {
          return renderSeatStack(state.cards_left[seat]);
        }
        return `<div class="seat-side-grid">${Array.from({length: Math.min(state.cards_left[seat], 12)}, () => renderCardBack("seat-side-open")).join("")}</div>`;
      }
      if (seat === 2) {
        return renderCardGroup(cards, "seat-open", true, "无牌").replace('card-group overlap-hand', 'card-group seat-top-group');
      }
      if (seat === 0) {
        return renderCardGroup(cards, "hand", true, "无牌");
      }
      return `<div class="seat-side-grid">${cards.map(card => renderPlayingCard(card, "seat-side-open")).join("")}</div>`;
    }

    function normalizeCards(cards) {
      return [...cards].sort().join("|");
    }

    function selectCard(card) {
      if (!latestState || latestState.current_player !== latestState.human_seat || latestState.is_game_over) {
        return;
      }
      if (selectedCards.includes(card)) {
        selectedCards = selectedCards.filter(item => item !== card);
      } else {
        selectedCards = [...selectedCards, card];
      }
      render(latestState);
    }

    function clearSelection() {
      selectedCards = [];
      if (latestState) render(latestState);
    }

    function buildIndexedActions(state) {
      return state.legal_actions.map((action, index) => ({...action, action_index: index}));
    }

    async function submitAction(action) {
      const previousState = latestState;
      const nextState = await api("/api/play", {
        method: "POST",
        body: JSON.stringify({action_index: action.action_index, turn_id: latestState.turn_id, action_id: action.action_id}),
      }).catch(error => alert(error.message));
      if (nextState) {
        selectedCards = [];
        await playMoveSequence(previousState, nextState);
      }
    }

    async function playSelected() {
      if (!latestState) return;
      if (!selectedCards.length) {
        alert("请先选择手牌");
        return;
      }
      const previousState = latestState;
      const humanSourceRect = getSelectedHandRect(selectedCards);
      const nextState = await api("/api/play-cards", {
        method: "POST",
        body: JSON.stringify({cards: selectedCards, turn_id: latestState.turn_id}),
      }).catch(error => alert(error.message));
      if (nextState) {
        await playMoveSequence(previousState, nextState, {humanSourceRect});
      }
    }

    async function passTurn() {
      if (!latestState) return;
      const passAction = buildIndexedActions(latestState).find(action => action.move_type === "PASS");
      if (!passAction) {
        alert("当前不能过牌");
        return;
      }
      await submitAction(passAction);
    }

    function hintAction() {
      if (!latestState) return;
      const hint = buildIndexedActions(latestState).find(action => action.move_type !== "PASS");
      if (!hint) {
        alert("当前没有可出的非过牌动作");
        return;
      }
      selectedCards = [...hint.cards];
      render(latestState);
    }

    function renderSelectedCards() {
      const node = document.getElementById("selected-summary");
      node.textContent = selectedCards.length ? `${selectedCards.length} 张` : "-";
    }

    function sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    function setAnimating(flag) {
      isAnimating = flag;
      window.__pdkAnimating = flag;
    }

    function getCanvasContext() {
      const canvas = document.getElementById("flight-canvas");
      const ratio = window.devicePixelRatio || 1;
      const width = window.innerWidth;
      const height = window.innerHeight;
      if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
        canvas.width = Math.floor(width * ratio);
        canvas.height = Math.floor(height * ratio);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return ctx;
    }

    function clearFlightCanvas() {
      const ctx = getCanvasContext();
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    }

    function clearLeadDisplay() {
      const node = document.getElementById("lead");
      if (node) {
        node.innerHTML = "";
      }
    }

    function getRectFromElement(element) {
      if (!element) return null;
      const rect = element.getBoundingClientRect();
      if (!rect.width || !rect.height) {
        return null;
      }
      return {
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
        right: rect.right,
        bottom: rect.bottom,
        centerX: rect.left + rect.width / 2,
        centerY: rect.top + rect.height / 2,
      };
    }

    function mergeRects(rects) {
      const valid = rects.filter(Boolean);
      if (!valid.length) return null;
      const left = Math.min(...valid.map(rect => rect.left));
      const top = Math.min(...valid.map(rect => rect.top));
      const right = Math.max(...valid.map(rect => rect.right));
      const bottom = Math.max(...valid.map(rect => rect.bottom));
      return {
        left,
        top,
        right,
        bottom,
        width: right - left,
        height: bottom - top,
        centerX: (left + right) / 2,
        centerY: (top + bottom) / 2,
      };
    }

    function getSeatElement(player) {
      if (player === 1) return document.getElementById("seat-left");
      if (player === 2) return document.getElementById("seat-top");
      if (player === 3) return document.getElementById("seat-right");
      return document.querySelector(".player-panel");
    }

    function getSelectedHandRect(cards) {
      const selected = cards.map(card => {
        const node = document.querySelector(`#hand-cards [data-card="${CSS.escape(card)}"]`);
        return getRectFromElement(node);
      });
      return mergeRects(selected);
    }

    function getMoveSourceRect(player, cards, preferredRect = null) {
      if (preferredRect) {
        return preferredRect;
      }
      if (player === 0) {
        return getSelectedHandRect(cards)
          || getRectFromElement(document.querySelector("#hand-cards .card-group"))
          || getRectFromElement(document.querySelector(".player-panel"));
      }
      const seat = getSeatElement(player);
      if (!seat) {
        return null;
      }
      return getRectFromElement(seat.querySelector(".seat-top-group, .seat-side-grid, .seat-stack, .card-group"))
        || getRectFromElement(seat);
    }

    function getSeatPassAnchor(player) {
      const seat = getSeatElement(player);
      const rect = getRectFromElement(seat);
      if (!rect) {
        return null;
      }
      if (player === 2) {
        return {left: rect.centerX, top: rect.bottom + 22};
      }
      if (player === 1) {
        return {left: rect.right + 28, top: rect.centerY};
      }
      if (player === 3) {
        return {left: rect.left - 28, top: rect.centerY};
      }
      return {left: rect.centerX, top: rect.top - 22};
    }

    function showSeatPass(player) {
      const overlay = document.getElementById("seat-pass-overlay");
      const anchor = getSeatPassAnchor(player);
      if (!overlay || !anchor) {
        return;
      }
      overlay.dataset.player = String(player);
      overlay.textContent = "PASS";
      overlay.style.left = `${anchor.left}px`;
      overlay.style.top = `${anchor.top}px`;
      overlay.classList.add("visible");
    }

    function hideSeatPass() {
      const overlay = document.getElementById("seat-pass-overlay");
      if (!overlay) {
        return;
      }
      overlay.dataset.player = "";
      overlay.classList.remove("visible");
      overlay.textContent = "";
    }

    function getLeadTargetRect() {
      return getRectFromElement(document.getElementById("lead"))
        || getRectFromElement(document.querySelector(".board-center-move"));
    }

    function shouldPauseForNextPlayer(nextPlayer, nextState) {
      return typeof nextPlayer === "number" && !nextState.is_game_over && nextPlayer !== nextState.human_seat;
    }

    function drawRoundedRect(ctx, x, y, width, height, radius) {
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + width - radius, y);
      ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
      ctx.lineTo(x + width, y + height - radius);
      ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
      ctx.lineTo(x + radius, y + height);
      ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
    }

    function drawCardFace(ctx, card, centerX, centerY, width, height, rotation) {
      const parsed = parseCard(card);
      const color = parsed.color === "red" ? "#d11f3a" : "#202733";
      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(rotation);
      ctx.translate(-width / 2, -height / 2);
      ctx.shadowColor = "rgba(0, 0, 0, 0.20)";
      ctx.shadowBlur = 14;
      ctx.shadowOffsetY = 6;
      drawRoundedRect(ctx, 0, 0, width, height, 12);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#1f2933";
      ctx.stroke();
      ctx.shadowColor = "transparent";
      ctx.fillStyle = color;
      ctx.font = "700 16px 'Segoe UI'";
      ctx.fillText(parsed.rank, 12, 22);
      ctx.font = "15px 'Segoe UI Symbol', 'Segoe UI'";
      ctx.fillText(parsed.symbol, 12, 40);
      ctx.font = "34px 'Segoe UI Symbol', 'Segoe UI'";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(parsed.symbol, width / 2, height / 2 + 2);
      ctx.restore();
    }

    function drawFlightFrame(ctx, cards, sourceRect, targetRect, progress) {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      if (!sourceRect || !targetRect) {
        return;
      }
      const baseWidth = 66;
      const baseHeight = 94;
      const spread = Math.max(18, Math.min(26, cards.length * 3));
      const arcHeight = Math.max(80, Math.min(180, Math.abs(targetRect.centerY - sourceRect.centerY) * 0.42 + 40));
      cards.forEach((card, index) => {
        const slot = index - (cards.length - 1) / 2;
        const startX = sourceRect.centerX + slot * spread * 0.72;
        const endX = targetRect.centerX + slot * spread;
        const startY = sourceRect.centerY + Math.abs(slot) * 3;
        const endY = targetRect.centerY + Math.abs(slot) * 2;
        const x = startX + (endX - startX) * progress;
        const y = startY + (endY - startY) * progress - Math.sin(progress * Math.PI) * arcHeight;
        const rotation = (-0.22 + slot * 0.06) * (1 - progress) + slot * 0.03 * progress;
        const scale = 0.92 + Math.sin(progress * Math.PI * 0.5) * 0.12;
        drawCardFace(ctx, card, x, y, baseWidth * scale, baseHeight * scale, rotation);
      });
    }

    async function animateCanvasMove(player, move, preferredRect = null) {
      if (!move || !move.cards || !move.cards.length) {
        return;
      }
      const sourceRect = getMoveSourceRect(player, move.cards, preferredRect);
      const targetRect = getLeadTargetRect();
      if (!sourceRect || !targetRect) {
        return;
      }
      const ctx = getCanvasContext();
      const started = performance.now();
      await new Promise(resolve => {
        function step(now) {
          const progress = Math.min((now - started) / AI_MOVE_FLY_MS, 1);
          drawFlightFrame(ctx, move.cards, sourceRect, targetRect, progress);
          if (progress < 1) {
            requestAnimationFrame(step);
            return;
          }
          resolve();
        }
        requestAnimationFrame(step);
      });
      drawFlightFrame(ctx, move.cards, sourceRect, targetRect, 1);
    }

    function cloneState(state) {
      return JSON.parse(JSON.stringify(state));
    }

    function removeCardsFromHand(cards, playedCards) {
      const remaining = [...(cards || [])];
      for (const card of playedCards || []) {
        const index = remaining.indexOf(card);
        if (index >= 0) {
          remaining.splice(index, 1);
        }
      }
      return remaining;
    }

    function buildInFlightPreviewState(baseState, entry) {
      const preview = cloneState(baseState);
      if (!entry || entry.move.move_type === "PASS") {
        return preview;
      }
      const nextHand = removeCardsFromHand(preview.all_hands[entry.player], entry.move.cards);
      preview.all_hands[entry.player] = nextHand;
      preview.cards_left[entry.player] = nextHand.length;
      if (entry.player === preview.human_seat) {
        preview.hand_cards = [...nextHand];
      }
      preview.current_player = entry.player;
      return preview;
    }

    function applyEntryToPreviewState(baseState, entry, nextPlayer) {
      const landed = entry.move.move_type === "PASS"
        ? cloneState(baseState)
        : buildInFlightPreviewState(baseState, entry);
      landed.history = [...baseState.history, entry];
      if (entry.move.move_type !== "PASS") {
        landed.lead_move = entry.move;
        landed.lead_player = entry.player;
      }
      if (typeof nextPlayer === "number") {
        landed.current_player = nextPlayer;
      }
      return landed;
    }

    async function playMoveSequence(previousState, nextState, options = {}) {
      if (!previousState || nextState.history.length < previousState.history.length) {
        render(nextState);
        return;
      }
      const newEntries = nextState.history.slice(previousState.history.length);
      if (!newEntries.length) {
        render(nextState);
        return;
      }

      setAnimating(true);
      let previewState = cloneState(previousState);
      try {
        for (let index = 0; index < newEntries.length; index += 1) {
          const entry = newEntries[index];
          const nextPlayer = index + 1 < newEntries.length
            ? newEntries[index + 1].player
            : nextState.current_player;
          if (entry.move.move_type === "PASS") {
            const passState = applyEntryToPreviewState(previewState, entry, nextPlayer);
            render(passState);
            showSeatPass(entry.player);
            if (shouldPauseForNextPlayer(nextPlayer, nextState)) {
              await sleep(AI_CHAIN_PAUSE_MS);
            }
            hideSeatPass();
            previewState = passState;
          } else {
            const inFlightState = buildInFlightPreviewState(previewState, entry);
            render(inFlightState);
            clearLeadDisplay();
            hideSeatPass();
            const preferredRect = entry.player === nextState.human_seat ? options.humanSourceRect || null : null;
            await animateCanvasMove(entry.player, entry.move, preferredRect);
            previewState = applyEntryToPreviewState(previewState, entry, nextPlayer);
            render(previewState);
            clearFlightCanvas();
            if (shouldPauseForNextPlayer(nextPlayer, nextState)) {
              await sleep(AI_CHAIN_PAUSE_MS);
            }
          }
        }
      } finally {
        hideSeatPass();
        clearFlightCanvas();
        setAnimating(false);
      }
      render(nextState);
    }

    function render(state) {
      latestState = state;
      selectedCards = selectedCards.filter(card => state.hand_cards.includes(card));
      const indexedActions = buildIndexedActions(state);
      const visibleHistory = state.history.filter(item => item.move.move_type !== "PASS");
      const hasNonPassAction = indexedActions.some(action => action.move_type !== "PASS");
      const hasPassAction = indexedActions.some(action => action.move_type === "PASS");
      document.getElementById("status").textContent = `Current turn: p${state.current_player + 1}`;
      document.getElementById("lead").innerHTML = state.lead_move
        ? renderCardGroup(state.lead_move.cards, "board", false, state.lead_move.text)
        : `<div class="muted">暂无出牌</div>`;
      const lastMove = visibleHistory.length ? visibleHistory[visibleHistory.length - 1] : null;
      document.getElementById("last-move").innerHTML = renderMoveSummary("最近出牌", lastMove);
      renderSelectedCards();

      const seatMap = {
        2: document.getElementById("seat-top"),
        1: document.getElementById("seat-left"),
        3: document.getElementById("seat-right"),
      };
      Object.entries(seatMap).forEach(([seat, node]) => {
        const seatNumber = Number(seat);
        const current = seatNumber === state.current_player;
        const score = state.scores ? state.scores[seatNumber] : 0;
        node.innerHTML = `<div class="seat-line">p${seatNumber + 1} [AI] | 剩余: ${state.cards_left[seat]} | 积分: ${score}${current ? " | 当前" : ""}</div>${renderSeatCards(seatNumber, state, revealHands)}`;
      });

      const hand = document.getElementById("hand-cards");
      hand.innerHTML = `<div class="card-group overlap-hand">${state.hand_cards.map(card => {
        const selectedClass = selectedCards.includes(card) ? " selected" : "";
        const parsed = parseCard(card);
        return `
          <div class="playing-card hand ${parsed.color}${selectedClass}" data-card="${escapeHtml(card)}" title="${escapeHtml(card)}">
            <svg class="playing-card-svg" viewBox="0 0 120 170" role="img" aria-label="${escapeHtml(card)}">
              <rect x="4" y="4" width="112" height="162" rx="14" fill="#ffffff" stroke="#1f2933" stroke-width="2"/>
              <text x="18" y="28" font-size="22" font-weight="700" fill="currentColor">${escapeHtml(parsed.rank)}</text>
              <text x="18" y="52" font-size="20" fill="currentColor">${parsed.symbol}</text>
              <text x="60" y="94" text-anchor="middle" dominant-baseline="middle" font-size="44" fill="currentColor">${parsed.symbol}</text>
              <g transform="translate(120 170) rotate(180)">
                <text x="18" y="28" font-size="22" font-weight="700" fill="currentColor">${escapeHtml(parsed.rank)}</text>
                <text x="18" y="52" font-size="20" fill="currentColor">${parsed.symbol}</text>
              </g>
            </svg>
          </div>
        `;
      }).join("")}</div>`;
      hand.querySelectorAll("[data-card]").forEach(node => {
        node.onclick = () => selectCard(node.getAttribute("data-card"));
      });
      document.getElementById("seat-bottom-title").textContent = `p1 · tfang [我]${state.current_player === 0 ? "[当前]" : ""} | 剩余: ${state.cards_left[0]} | 积分: ${state.scores ? state.scores[0] : 0}`;
      const toggle = document.getElementById("toggle-hand-visibility");
      toggle.textContent = revealHands ? "切换牌背模式" : "显示四家明牌";
      document.getElementById("play-selected").disabled =
        isAnimating || state.current_player !== 0 || state.is_game_over || selectedCards.length === 0;
      document.getElementById("pass-action").disabled =
        isAnimating || state.current_player !== 0 || state.is_game_over || !hasPassAction;
      document.getElementById("hint-action").disabled =
        isAnimating || state.current_player !== 0 || state.is_game_over || !hasNonPassAction;
      document.getElementById("clear-selection").disabled = isAnimating || selectedCards.length === 0;
    }

    async function refresh() {
      const state = await api("/api/state");
      await playMoveSequence(latestState, state);
    }

    document.getElementById("new-game").onclick = async () => {
      const state = await api("/api/new-game", {method: "POST"});
      selectedCards = [];
      await playMoveSequence(latestState, state);
    };

    document.getElementById("toggle-hand-visibility").onclick = async () => {
      revealHands = !revealHands;
      const state = await api("/api/state");
      render(state);
    };

    document.getElementById("play-selected").onclick = playSelected;
    document.getElementById("pass-action").onclick = passTurn;
    document.getElementById("clear-selection").onclick = clearSelection;
    document.getElementById("hint-action").onclick = hintAction;

    refresh().catch(async () => {
      const state = await api("/api/new-game", {method: "POST"});
      await playMoveSequence(latestState, state);
    });
  </script>
</body>
</html>
"""


class WebApp:
    def __init__(self, checkpoint_path: Path) -> None:
        self.session = WebGameSession(checkpoint_path)

    def __call__(self, environ, start_response):
        method = environ["REQUEST_METHOD"]
        path = environ.get("PATH_INFO", "/")

        try:
            if method == "GET" and path == "/":
                return _html_response(start_response, INDEX_HTML)
            if method == "GET" and path == "/api/state":
                return _json_response(start_response, 200, self.session.get_state())
            if method == "POST" and path == "/api/new-game":
                return _json_response(start_response, 200, self.session.new_game())
            if method == "POST" and path == "/api/play":
                payload = _read_json(environ)
                state = self.session.play(
                    int(payload["action_index"]),
                    turn_id=int(payload["turn_id"]),
                    action_id=str(payload["action_id"]),
                )
                return _json_response(start_response, 200, state)
            if method == "POST" and path == "/api/play-cards":
                payload = _read_json(environ)
                state = self.session.play_cards(
                    [str(card) for card in payload.get("cards", [])],
                    turn_id=int(payload["turn_id"]),
                )
                return _json_response(start_response, 200, state)
            return _json_response(start_response, 404, {"error": "not found"})
        except IndexError:
            return _json_response(start_response, 400, {"error": "invalid action index"})
        except PermissionError as error:
            return _json_response(start_response, 409, {"error": str(error)})
        except ValueError as error:
            return _json_response(start_response, 400, {"error": str(error)})


def create_app(checkpoint_path: str | Path):
    return WebApp(Path(checkpoint_path))


def _json_response(start_response, status_code: int, payload: dict):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    reason = {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        409: "Conflict",
    }.get(status_code, "OK")
    start_response(
        f"{status_code} {reason}",
        [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))],
    )
    return [body]


def _html_response(start_response, html: str):
    body = html.encode("utf-8")
    start_response(
        "200 OK",
        [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))],
    )
    return [body]


def _read_json(environ) -> dict:
    length = int(environ.get("CONTENT_LENGTH") or "0")
    raw = environ["wsgi.input"].read(length) if length > 0 else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))
